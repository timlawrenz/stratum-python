# Plan: examples/bulk_process.py — concurrent bulk image processing

## Goal

Replace the serial `process_local_folder.py` with a new example that shows
a third party how to correctly process hundreds or thousands of images using
the stratum-python primitives.

## Current state / problems with process_local_folder.py

- `analyze_file()` blocks per image: submit → poll → fetch → save, then move on.
  50 images × ~10s = serial 8+ minute run.
- No concurrency. One image in flight at a time.
- No checkpoint. Crash at image 800 → restart from 0.
- No per-image error isolation (the except swallows and continues, but job
  failures still block the slot).
- Hard-coded to staging URL and a fixed operation set.
- No progress feedback beyond a single print per image.

---

## Proposed approach

Use `concurrent.futures.ThreadPoolExecutor` with a bounded semaphore so at
most N jobs are in flight simultaneously (default: 20). Decouple submission
from collection.

Pattern:
  1. Walk input directory, collect image paths.
  2. Load checkpoint file (JSON) listing already-completed stems → skip them.
  3. Submit jobs up to `--concurrency` in-flight via the thread pool.
  4. Each worker thread: submit_file → jobs.wait → jobs.results → save outputs.
  5. On success write stem to checkpoint file (append-safe).
  6. On failure log error, continue.
  7. Final summary line: N succeeded / M failed / K skipped.

The thread pool keeps `--concurrency` workers busy at all times. The GIL is not
a concern here because each thread spends almost all of its time in httpx I/O
(network blocked, not CPU).

---

## Script: examples/bulk_process.py

### CLI interface

```
python bulk_process.py INPUT_DIR OUTPUT_DIR [options]

  --concurrency INT     Max parallel jobs in flight (default: 20)
  --base-url URL        API base URL (default: https://stratum.pi216.ai)
  --job-timeout FLOAT   Per-job timeout in seconds (default: 300)
  --operations JSON     JSON string or @file.json overriding the default operation set
  --extensions LIST     Comma-separated extensions (default: jpg,jpeg,png,webp)
  --dry-run             List images that would be processed, do not submit
```

`STRATUM_API_KEY` env var is required (same as existing example).

### Operation defaults

Same full set as process_local_folder.py (all ops enabled) but loaded from a
constant dict so it can be overridden via `--operations`.

### Output structure

Same as process_local_folder.py:
  OUTPUT_DIR/{stem}/clip.npy
  OUTPUT_DIR/{stem}/caption.txt
  OUTPUT_DIR/{stem}/depth.npy
  ... etc.

### Checkpoint file

JSON object: `{ "completed": ["stem1", "stem2", ...] }`
Written atomically: write to .tmp then os.replace().
Thread-safe via a threading.Lock around reads/writes.

### Error file

`OUTPUT_DIR/.errors.jsonl` — one JSON object per line:
  `{"stem": "img001", "error": "JobFailed: ...", "timestamp": "..."}`

---

## Step-by-step implementation plan

### Step 1 — Skeleton

Create `examples/bulk_process.py` with:
- argparse setup (all flags above)
- STRATUM_API_KEY env check
- Image discovery (pathlib walk, extension filter)
- Dry-run early exit

### Step 2 — Done detection via sentinel file

No checkpoint class needed. Use the output folder itself:

  is_done   → `(output_dir / stem / ".done").exists()`
  mark_done → `(output_dir / stem / ".done").touch()` written as the very last
              step inside `save_results()`, after all .npy / .txt files are flushed.

Rationale: the output is its own checkpoint. A third party inspecting the
output directory immediately understands the state. Zero extra state to lose.

Partial-write safety: if the process crashes mid-save the .done file is never
written, so the directory is re-processed on the next run and outputs are
overwritten cleanly.

### Step 3 — Error logger

Function `log_error(errors_path, stem, exc, lock)`:
  - Acquire lock, open in append mode, write JSON line, flush, close.

### Step 4 — Worker function

```python
def process_one(img_path, output_dir, client, ops, job_timeout,
                checkpoint, errors_path, err_lock, progress):
    stem = img_path.stem
    img_out = output_dir / stem
    img_out.mkdir(parents=True, exist_ok=True)

    try:
        job = client.jobs.submit_file(
            str(img_path),
            whole_image=ops.get("whole_image"),
            prominent_person=ops.get("prominent_person"),
            prominent_face=ops.get("prominent_face"),
        )
        job = client.jobs.wait(job.job_id, timeout=job_timeout)
        results = client.jobs.results(job.job_id, _job=job)
        save_results(results, img_out)
        checkpoint.mark_done(stem)
    except Exception as exc:
        log_error(errors_path, stem, exc, err_lock)
    finally:
        progress.update()  # thread-safe counter increment + print
```

### Step 5 — save_results()

Extract the npy/txt saving logic from process_local_folder.py into a standalone
`save_results(results: JobResults, out_dir: Path)` function. Handle missing
sections/ops gracefully (check `if "clip" in wi`). Same logic, just refactored.

For prominent_person and prominent_face: mirror what whole_image saves but into
`out_dir/prominent_person/` and `out_dir/prominent_face/` subdirs.

### Step 6 — Progress tracker

Class `Progress`:
  - Tracks submitted, done, failed with a `threading.Lock`
  - `update(success: bool)` prints a line like:
    `[42/1000] 41 ok | 1 failed | 957 remaining`
  - Uses `\r` overwrite for compact terminal output (or plain newline if not a tty)

### Step 7 — Main entrypoint

```python
with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
    futures = {
        pool.submit(process_one, img, ...): img
        for img in images
        if not checkpoint.is_done(img.stem)
    }
    for fut in as_completed(futures):
        pass  # errors already handled inside process_one
```

Note: all error handling is inside `process_one` so futures never raise.
`as_completed` is used only to keep the main thread alive until all are done.

### Step 8 — Final summary

Print:
```
Done. 998 succeeded / 2 failed / 0 skipped (checkpoint).
Failed images logged to OUTPUT_DIR/.errors.jsonl
```

---

## Files to create / change

| File | Action |
|---|---|
| `examples/bulk_process.py` | Create (new) |
| `examples/process_local_folder.py` | No change (keep as simple single-threaded reference) |

No changes to library code. This is example-layer only.

---

## Testing / validation

1. Smoke test with 3 real images, concurrency=2, confirm outputs written and
   checkpoint populated.
2. Interrupt mid-run (Ctrl-C), re-run — confirm already-done images are skipped.
3. Inject a bad image path to confirm error is written to .errors.jsonl and run
   continues.
4. --dry-run confirms image list printed, nothing submitted, no output dirs created.
5. Run with --no-checkpoint confirms all images processed even if checkpoint exists.

No automated test file is planned (no test suite in the repo). Manual smoke
tests suffice for an example script.

---

## Risks and tradeoffs

- **Concurrency=20 default** is a guess. The API and Vast.ai worker pool set
  the real ceiling. If the queue gets saturated, jobs will wait server-side
  anyway. A user can tune down with --concurrency.

- **Thread pool vs asyncio**: ThreadPoolExecutor is simpler for a third-party
  example. No async/await, no event loop, no nest_asyncio. httpx is sync here.
  An async version would be faster at extreme scale (1000+) but adds cognitive
  overhead for an example. Call this out in the docstring.

- **Checkpoint is stem-based**: if two files have the same stem but different
  extensions, the second will be skipped. Acceptable for an example; note in
  docstring.

- **No rate-limit back-pressure from the client side**: `sync_request_with_retry`
  already handles 429 with Retry-After. The thread pool naturally limits
  submission rate by `--concurrency`. Should be fine.

- **Memory**: all results are decoded and saved to disk immediately inside the
  worker. No accumulation in memory. Safe for thousands of images.

---

## Open questions

- Should prominent_person / prominent_face outputs go into subdirs or flat with
  a prefix (e.g. `pp_clip.npy`)? Plan assumes subdirs. Confirm before implementing.
- Should the script support image URLs as input (a --urls-file flag)?
  Out of scope for now; submit_file is the primary use case shown here.
