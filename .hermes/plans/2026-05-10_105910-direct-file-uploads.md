# Plan: Extend Stratum API and Client for Direct File Uploads

## Goal
Extend the `stratum-api` control plane and the `stratum-python` SDK to allow direct file uploads. We will implement both standard multipart/form-data uploads and Base64-encoded JSON payloads (modern UUEncoding) for maximum client flexibility.

## Current Context / Assumptions
- The API (`stratum-api`) uses FastAPI and currently has an `/analyze_image_upload` endpoint that accepts multipart form data. However, it's not well integrated into the standard RESTful `/jobs` namespace.
- The `JobRequest` schema only supports an `image_url` string.
- The Python client (`stratum-python`) has methods like `submit` and `analyze` which require passing an `image_url`.
- R2/S3 is available. Files sent directly to the API must be stored in R2, converting the job's target into an `r2://` URI before it enters the Redis job queue.

## Proposed Approach
1. **API side**:
   - Clean up the `/analyze_image_upload` route by adding a clean REST alias or renaming it to `POST /jobs/upload`.
   - Update `JobRequest` to optionally accept `image_base64`. When the API receives base64 payload on `POST /jobs`, it decodes the image, uploads it to R2, and replaces it with an `r2://` URL before queueing.
2. **Client side**:
   - Update `stratum-python` `AnalyzeImageRequest` models to mirror the new API schema.
   - Add `submit_file()` / `analyze_file()` functions to send local files via multipart/form-data to `/jobs/upload`.
   - Add an `image_base64` parameter to `submit()` / `analyze()` to allow embedding file data directly into the JSON.

## Step-by-Step Plan

### Phase 1: API Updates (`../stratum-api`)
1. **Schema Updates (`src/stratum_api/models/schemas.py`)**:
   - Add `image_base64: str | None = None` to `JobRequest`.
   - Relax `image_url` to be optional (`str | None = None`).
   - Add a model validator ensuring exactly one of `image_url` or `image_base64` is provided.
2. **Route Updates (`src/stratum_api/main.py`)**:
   - **Base64**: In the `POST /jobs` route, check for `body.image_base64`. If present, base64-decode the bytes, determine the MIME type (from headers or magic bytes), upload the bytes to R2 (using `generate_upload_url` and `upload_bytes`), and map the resulting R2 key to `body.image_url` so the downstream workers function normally.
   - **Multipart**: Add a `POST /jobs/upload` route that wraps the logic inside the existing `/analyze_image_upload` route, keeping the REST pattern clean.
3. **API Tests**:
   - Validate base64 payload decode + queueing.
   - Validate multipart form submission to `/jobs/upload`.

### Phase 2: Client Updates (`./stratum-python`)
1. **Models (`src/stratum/models.py`)**:
   - Update `AnalyzeImageRequest` to make `image_url` optional and add `image_base64: str | None`.
2. **Client Multipart Support (`src/stratum/client.py`)**:
   - Add `_JobsNamespace.submit_file(file_path: str | Path, ...)`:
     - Open the file, build the `files={"image_file": ...}` and `data={"request_json": json.dumps(req_dict)}` maps.
     - Call `self._client._request("POST", "/jobs/upload", files=files, data=data)`.
   - Add a matching `StratumClient.analyze_file()` helper.
3. **Client Base64 Support (`src/stratum/client.py`)**:
   - Update `submit()` and `analyze()` signatures to accept `image_base64: str | bytes | None`. If raw bytes are passed, base64 encode them before passing them to the model payload.
4. **Client Tests**:
   - Add tests inside `tests/` to verify payloads for both `submit_file` (multipart structures) and `submit(image_base64=...)`.

## Files Likely to Change
**stratum-api**:
- `src/stratum_api/models/schemas.py`
- `src/stratum_api/main.py`
- `tests/test_jobs.py` (or new test file)

**stratum-python**:
- `src/stratum/models.py`
- `src/stratum/client.py`
- `tests/test_client.py`

## Tests / Validation
- **Local execution**: Both `stratum-api` and `stratum-python` use `pytest`, `ruff`, and `mypy`. Ensure these run successfully on both repositories before committing.
  - `cd ../stratum-api && ruff check . && mypy . && pytest`
  - `cd . && ruff check . && mypy . && pytest`

## Risks, Tradeoffs, and Open Questions
- **Payload Size Constraints**: Base64 strings bloat payload size by ~33%. Very large images could run afoul of HTTP `413 Payload Too Large` limits configured on the load balancer or inside FastAPI.
- **Queue limits**: Base64 strings MUST be stripped off the job payload after the R2 upload, so they are not serialized into the Redis queue (which limits size and impacts memory).
- **Security**: For multipart uploads and base64 parsing, the server should validate image signatures (magic numbers) before blind-uploading to R2 to prevent arbitrary code/file drops.