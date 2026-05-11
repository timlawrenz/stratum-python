"""
bulk_process.py — concurrent bulk image processing with stratum-python

Processes hundreds or thousands of local images in parallel using a thread
pool.  Each image is submitted as a file upload, waited on, and its results
saved to disk.  The output directory itself serves as the checkpoint: a
`.done` sentinel file inside each image's output folder marks it as complete,
so interrupted runs resume cleanly without reprocessing finished images.

Usage
-----
    export STRATUM_API_KEY=sk_...
    python bulk_process.py /path/to/images /path/to/output [options]

    --concurrency INT     Max parallel jobs in flight (default: 20)
    --base-url URL        API base URL (default: https://stratum.pi216.ai)
    --job-timeout FLOAT   Per-job timeout in seconds (default: 300)
    --operations JSON     JSON string or @file.json overriding the default
                          operation set.  Must be a dict with keys
                          "whole_image", "prominent_person", "prominent_face".
    --extensions LIST     Comma-separated file extensions to process
                          (default: jpg,jpeg,png,webp)
    --dry-run             Print images that would be processed, then exit.

Notes
-----
- Uses concurrent.futures.ThreadPoolExecutor (not asyncio).  This keeps the
  example straightforward.  At extreme scale (5000+ images) an async client
  would reduce thread overhead, but for most workloads 20 threads is plenty.
- Duplicate stems across extensions: if foo.jpg and foo.png both exist, only
  the first to finish will mark .done; the second run will skip it.  Ensure
  unique stems if this matters.
- Errors are appended line-by-line to OUTPUT_DIR/.errors.jsonl so you can
  inspect and retry specific images without re-running the whole batch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stratum import StratumClient
from stratum.results import JobResults

# ── Default operation set ────────────────────────────────────────────────────

DEFAULT_OPERATIONS: dict[str, dict[str, bool]] = {
    "whole_image": {
        "clip": True,
        "dino_v2": True,
        "dinov3_cls": True,
        "dinov3_full": True,
        "caption": True,
        "t5": True,
        "pixel": True,
    },
    "prominent_person": {
        "clip": True,
        "dino_v2": True,
        "dinov3_cls": True,
        "dinov3_full": True,
        "caption": True,
        "t5": True,
        "pose": True,
        "seg": True,
        "depth": True,
        "normal": True,
    },
    "prominent_face": {
        "clip": True,
        "dino_v2": True,
        "dinov3_cls": True,
        "dinov3_full": True,
        "caption": True,
        "t5": True,
        "pose": True,
        "seg": True,
        "depth": True,
        "normal": True,
    },
}

# ── Output helpers ───────────────────────────────────────────────────────────


def _save_section(section_results, out_dir: Path) -> None:
    """Save all recognised outputs from one section (whole_image / person / face)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if "pixel" in section_results and section_results.pixel.parsed:
        np.save(out_dir / "pixel.npy", section_results.pixel.parsed.to_numpy())

    if "caption" in section_results and section_results.caption.parsed:
        (out_dir / "caption.txt").write_text(section_results.caption.parsed.text, encoding="utf-8")

    if "depth" in section_results and section_results.depth.parsed:
        np.save(out_dir / "depth.npy", section_results.depth.parsed.to_numpy())

    if "normal" in section_results and section_results.normal.parsed:
        np.save(out_dir / "normal.npy", section_results.normal.parsed.to_numpy())

    if "seg" in section_results and section_results.seg.parsed:
        np.save(out_dir / "seg.npy", section_results.seg.parsed.to_numpy())

    if "pose" in section_results and section_results.pose.parsed:
        arr = np.array(
            [[kp.x, kp.y, kp.confidence] for kp in section_results.pose.parsed.keypoints],
            dtype=np.float32,
        )
        np.save(out_dir / "pose.npy", arr)

    if "clip" in section_results and section_results.clip.parsed:
        np.save(out_dir / "clip.npy",
                np.array(section_results.clip.parsed.embedding, dtype=np.float32))

    if "dino_v2" in section_results and section_results.dino_v2.parsed:
        np.save(out_dir / "dino_v2.npy",
                np.array(section_results.dino_v2.parsed.embedding, dtype=np.float32))

    if "dinov3_cls" in section_results and section_results.dinov3_cls.parsed:
        np.save(out_dir / "dinov3_cls.npy",
                np.array(section_results.dinov3_cls.parsed.embedding, dtype=np.float32))

    if "dinov3_full" in section_results and section_results.dinov3_full.parsed:
        np.save(out_dir / "dinov3_patches.npy",
                np.array(section_results.dinov3_full.parsed.patches, dtype=np.float32))

    if "t5" in section_results and section_results.t5.parsed:
        hidden, mask = section_results.t5.parsed.to_numpy()
        np.save(out_dir / "t5_hidden.npy", hidden)
        np.save(out_dir / "t5_mask.npy", mask)


def save_results(results: JobResults, img_out: Path) -> None:
    """
    Persist all result sections to disk under img_out/.

    Layout:
        img_out/                  ← whole_image outputs live here
            clip.npy
            caption.txt
            ...
        img_out/prominent_person/
            clip.npy
            pose.npy
            ...
        img_out/prominent_face/
            clip.npy
            ...
        img_out/.done             ← sentinel written last; used by skip logic
    """
    _save_section(results.whole_image, img_out)
    _save_section(results.prominent_person, img_out / "prominent_person")
    _save_section(results.prominent_face, img_out / "prominent_face")

    # Sentinel written last — guarantees partial writes are retried on re-run.
    (img_out / ".done").touch()


# ── Progress tracker ─────────────────────────────────────────────────────────


class Progress:
    """Thread-safe progress reporter."""

    def __init__(self, total: int, skipped: int) -> None:
        self._total = total
        self._skipped = skipped
        self._done = 0
        self._failed = 0
        self._lock = threading.Lock()
        self._is_tty = sys.stderr.isatty()

    def update(self, success: bool) -> None:
        with self._lock:
            self._done += 1
            if not success:
                self._failed += 1
            completed = self._done
            failed = self._failed
            total = self._total

        ok = completed - failed
        remaining = total - completed
        line = f"[{completed}/{total}] {ok} ok | {failed} failed | {remaining} remaining"
        if self._is_tty:
            sys.stderr.write(f"\r{line}   ")
            sys.stderr.flush()
        else:
            sys.stderr.write(line + "\n")

    @property
    def counts(self) -> tuple[int, int, int]:
        """Return (succeeded, failed, skipped)."""
        with self._lock:
            return self._done - self._failed, self._failed, self._skipped


# ── Error logger ─────────────────────────────────────────────────────────────


def log_error(errors_path: Path, stem: str, exc: Exception, lock: threading.Lock) -> None:
    entry = json.dumps({
        "stem": stem,
        "error": f"{type(exc).__name__}: {exc}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    with lock:
        with errors_path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n")


# ── Worker ───────────────────────────────────────────────────────────────────


def process_one(
    img_path: Path,
    output_dir: Path,
    client: StratumClient,
    ops: dict[str, Any],
    job_timeout: float,
    errors_path: Path,
    err_lock: threading.Lock,
    progress: Progress,
) -> None:
    stem = img_path.stem
    img_out = output_dir / stem
    success = False
    try:
        img_out.mkdir(parents=True, exist_ok=True)
        job = client.jobs.submit_file(
            str(img_path),
            whole_image=ops.get("whole_image"),
            prominent_person=ops.get("prominent_person"),
            prominent_face=ops.get("prominent_face"),
        )
        job = client.jobs.wait(job.job_id, timeout=job_timeout)
        results = client.jobs.results(job.job_id, _job=job)
        save_results(results, img_out)
        success = True
    except Exception as exc:
        log_error(errors_path, stem, exc, err_lock)
    finally:
        progress.update(success)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Process a folder of images in parallel via the Stratum API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input_dir", help="Directory containing images to process.")
    p.add_argument("output_dir", help="Directory to write per-image results into.")
    p.add_argument("--concurrency", type=int, default=20,
                   help="Max parallel jobs in flight (default: 20).")
    p.add_argument("--base-url", default="https://stratum.pi216.ai",
                   help="Stratum API base URL.")
    p.add_argument("--job-timeout", type=float, default=300.0,
                   help="Per-job timeout in seconds (default: 300).")
    p.add_argument("--operations",
                   help="JSON string or @path.json overriding the default operation set.")
    p.add_argument("--extensions", default="jpg,jpeg,png,webp",
                   help="Comma-separated extensions to include (default: jpg,jpeg,png,webp).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print images that would be processed, then exit.")
    return p.parse_args()


def load_operations(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return DEFAULT_OPERATIONS
    if raw.startswith("@"):
        path = Path(raw[1:])
        text = path.read_text(encoding="utf-8")
    else:
        text = raw
    ops = json.loads(text)
    if not isinstance(ops, dict):
        raise ValueError("--operations must be a JSON object with keys whole_image / prominent_person / prominent_face")
    return ops


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("STRATUM_API_KEY")
    if not api_key:
        print("Error: STRATUM_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    ops = load_operations(args.operations)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.is_dir():
        print(f"Error: {in_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    exts = {f".{e.lstrip('.')}" for e in args.extensions.split(",")}
    images = sorted(
        f for f in in_dir.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )

    if not images:
        print(f"No images found in {in_dir} with extensions {exts}.")
        sys.exit(0)

    # Determine which images are already done (sentinel file present).
    pending = [img for img in images if not (out_dir / img.stem / ".done").exists()]
    skipped = len(images) - len(pending)

    print(f"Found {len(images)} images: {len(pending)} to process, {skipped} already done.")

    if args.dry_run:
        for img in pending:
            print(f"  would process: {img.name}")
        sys.exit(0)

    if not pending:
        print("Nothing to do.")
        sys.exit(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / ".errors.jsonl"
    err_lock = threading.Lock()
    progress = Progress(total=len(pending), skipped=skipped)

    client = StratumClient(
        api_key=api_key,
        base_url=args.base_url,
        job_timeout=args.job_timeout,
    )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                process_one,
                img, out_dir, client, ops, args.job_timeout,
                errors_path, err_lock, progress,
            ): img
            for img in pending
        }
        for _ in as_completed(futures):
            pass  # all handling is inside process_one; we just drain here

    if sys.stderr.isatty():
        sys.stderr.write("\n")

    succeeded, failed, _ = progress.counts
    print(f"\nDone. {succeeded} succeeded / {failed} failed / {skipped} skipped (already done).")
    if failed:
        print(f"Failed images logged to {errors_path}")

    client.close()


if __name__ == "__main__":
    main()
