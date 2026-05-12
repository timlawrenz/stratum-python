import httpx
import os
client = httpx.Client()
with open("test.jpg", "wb") as f:
    f.write(b"fake image data")

with open("test.jpg", "rb") as f:
    files = {"image_file": ("test.jpg", f, "image/jpeg")}
    data = {"request_json": '{"clip": true}'}
    resp = client.post("https://stratum-staging.pi216.ai/jobs/upload", 
        files=files, 
        data=data,
        headers={"Authorization": f"Bearer {os.environ.get('STRATUM_API_KEY')}"}
    )
    print(resp.status_code)
    print(resp.text)
