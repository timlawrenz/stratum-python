# stratum-python

Python SDK for the [Stratum](https://stratum.pi216.ai) image enrichment API.

```
pip install stratum
```

## Quick start

```python
from stratum import StratumClient

client = StratumClient(api_key="sk_...")

# Option 1: Provide a public URL
results = client.analyze(
    image_url="https://example.com/photo.jpg",
    whole_image={"clip": True},
    prominent_person={"pose": True, "seg": True},
)

# Option 2: Upload a local file directly (handles multipart form upload automatically)
results = client.analyze_file(
    file_path="/path/to/local/photo.jpg",
    whole_image={"clip": True},
    prominent_person={"pose": True, "seg": True},
)

# Option 3: Embed Base64 directly into the JSON request
import base64
with open("/path/to/local/photo.jpg", "rb") as f:
    b64_img = base64.b64encode(f.read()).decode("utf-8")

results = client.analyze(
    image_base64=b64_img,
    whole_image={"clip": True},
)

# Access typed results
clip = results.whole_image.clip
print(clip.parsed.embedding[:5])   # [0.12, -0.45, ...]
print(clip.parsed.dimensions)       # 512

det = results.prominent_person.bbox
print(det.parsed.detected)          # True
print(det.parsed.bbox)              # BBox(x1=341, y1=34, x2=1365, y2=2880)
```

## Async

```python
from stratum import AsyncStratumClient

async with AsyncStratumClient(api_key="sk_...") as client:
    # URL submission
    results = await client.analyze(
        image_url="https://example.com/photo.jpg",
        whole_image={"clip": True},
    )
    
    # Direct file upload
    results = await client.analyze_file(
        file_path="/path/to/local/photo.jpg",
        whole_image={"clip": True},
    )
```

## Step-by-step control

```python
# Submit without waiting
job = client.jobs.submit(
    image_url="https://example.com/photo.jpg", # Can also use image_base64=...
    whole_image={"clip": True},
    prominent_person={"seg": True},
)

# Or submit a local file without waiting
job = client.jobs.submit_file(
    file_path="/path/to/local/photo.jpg",
    whole_image={"clip": True},
)

print(job.job_id, job.status)  # UUID "queued"

# Wait for completion
job = client.jobs.wait(job.job_id, timeout=120)

# Download parsed results
results = client.jobs.results(job.job_id)
```

## Batch processing

```python
# Submit many images with the same operations
jobs = client.batch.submit(
    image_urls=["url1", "url2", "url3"],
    whole_image={"clip": True, "dino_v2": True},
    prominent_face={"dinov3_cls": True},
)

# Wait for all with progress
all_results = client.batch.wait_all(
    jobs,
    timeout=300,
    on_progress=lambda done, total: print(f"{done}/{total}"),
)
```

## Available operations

| Operation | Output | Credits |
|-----------|--------|---------|
| `embed_clip_vit_b_32` | 512-d semantic embedding | 1 |
| `embed_dino_v2` | 768-d visual embedding | 1 |
| `embed_dino_v3_cls` | 1024-d DINOv3 CLS token | 1 |
| `embed_dino_v3_full` | CLS + spatial patches | 2 |
| `detect_bounding_box` | Person/face bbox | 1 |
| `extract_pose` | 133 whole-body keypoints | 3 |
| `segment_body` | 28-class body segmentation | 5 |
| `estimate_depth` | Monocular depth map | 5 |
| `estimate_normals` | XYZ surface normals | 5 |
| `caption_image` | Dense image caption | 2 |
| `encode_t5` | T5-Large text encoding | 2 |

```python
# List operations
from stratum import list_operations, estimate_credits

ops = list_operations()
cost = estimate_credits(AnalyzeImageRequest(image_url="", whole_image={"clip": True}, prominent_person={"seg": True}))  # 7
```

## Typed results

Results are automatically parsed into typed objects:

- `EmbeddingResult` — `.embedding`, `.dimensions`, `.to_numpy()`
- `DetectionResult` — `.detected`, `.bbox`, `.confidence`
- `PoseResult` — `.keypoints` (list of `Keypoint`), `.num_keypoints`
- `SegmentationResult` — `.mask_base64`, `.shape`, `.classes`, `.to_numpy()`
- `PixelResult` — `.pixels_base64`, `.shape`, `.dtype`, `.to_numpy()`
- `DepthResult` — `.depth_base64`, `.shape`, `.to_numpy()`
- `NormalsResult` — `.normals_base64`, `.shape`, `.to_numpy()`
- `CaptionResult` — `.text`
- `T5Result` — `.hidden_base64`, `.hidden_shape`, `.to_numpy()`

Install with numpy support for array conversion:
```
pip install "stratum[numpy]"
```

## Webhook verification

```python
from stratum import verify_signature

is_valid = verify_signature(
    payload=request.body,
    signature=request.headers["X-Stratum-Signature"],
    secret="whsec_...",
)
```

## Error handling

```python
from stratum import (
    StratumError,
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    JobFailedError,
    JobTimeoutError,
)

try:
    results = client.analyze(image_url=url, whole_image={"clip": True})
except AuthenticationError:
    print("Bad API key")
except InsufficientCreditsError:
    print("Buy more credits")
except RateLimitError as e:
    print(f"Slow down, retry in {e.retry_after}s")
except JobFailedError as e:
    print(f"Job {e.job_id} failed: {e.error_message}")
except JobTimeoutError as e:
    print(f"Timed out after {e.timeout}s")
```

## Configuration

```python
client = StratumClient(
    api_key="sk_...",
    base_url="https://stratum.pi216.ai",  # default
    timeout=30.0,           # HTTP request timeout
    max_retries=3,          # retry on 429/5xx
    poll_interval=2.0,      # initial poll interval
    poll_max_interval=30.0, # max poll interval (exponential backoff)
    job_timeout=300.0,      # default wait timeout
)
```

## License

MIT
