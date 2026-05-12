import httpx
import os

client = httpx.Client()
with open("test.jpg", "wb") as f:
    f.write(b"fake image data")

with open("test.jpg", "rb") as f:
    # Key change: explicitly read the file into memory or use httpx syntax
    files = {"image_file": ("test.jpg", f, "image/jpeg")}
    data = {"request_json": '{"whole_image": {"clip": true}}'}
    
    resp = client.post(
        "https://stratum-staging.pi216.ai/jobs/upload", 
        files=files, 
        data=data,
        headers={"Authorization": f"Bearer {os.environ.get('STRATUM_API_KEY')}"}
    )
    print("Status:", resp.status_code)
    print("Response:", resp.text)
