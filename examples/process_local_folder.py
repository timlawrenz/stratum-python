import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from stratum import StratumClient


def process_local_directory(input_folder: str, output_folder: str, api_key: str):
    """
    Scans a directory for images, sends them to Stratum API via direct upload,
    requests the full analysis payload, and saves the parsed enrichment results
    in a structure matching stratum-hq.
    """
    client = StratumClient(
        api_key=api_key,
        # Using staging URL as an example; point to local if running the API locally
        base_url="https://stratum-staging.pi216.ai"
    )

    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    images = [f for f in in_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    print(f"Found {len(images)} images in {input_folder}\n")

    for img_path in images:
        # Create a dedicated output subdirectory for this image just like stratum-hq
        # We strip the extension to get the clean image ID
        img_id = img_path.stem
        img_out_dir = out_dir / img_id
        img_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Uploading and analyzing: {img_path.name}")
        try:
            results = client.analyze_file(
                file_path=str(img_path),
                whole_image={
                    "clip": True,
                    "dino_v2": True,
                    "dinov3_cls": True,
                    "dinov3_full": True,
                    "caption": True,
                    "t5": True,
                    "pixel": True
                },
                prominent_person={
                    "clip": True,
                    "dino_v2": True,
                    "dinov3_cls": True,
                    "dinov3_full": True,
                    "caption": True,
                    "t5": True,
                    "pose": True,
                    "seg": True,
                    "depth": True,
                    "normal": True
                },
                prominent_face={
                    "clip": True,
                    "dino_v2": True,
                    "dinov3_cls": True,
                    "dinov3_full": True,
                    "caption": True,
                    "t5": True,
                    "pose": True,
                    "seg": True,
                    "depth": True,
                    "normal": True
                }
            )

            wi = results.whole_image

            # Recreate stratum-hq file structure by decoding the API results back to disk
            
            if "pixel" in wi and wi.pixel.parsed:
                np.save(img_out_dir / "pixel.npy", wi.pixel.parsed.to_numpy())

            if "caption" in wi and wi.caption.parsed:
                with open(img_out_dir / "caption.txt", "w") as f:
                    f.write(wi.caption.parsed.text)

            if "depth" in wi and wi.depth.parsed:
                np.save(img_out_dir / "depth.npy", wi.depth.parsed.to_numpy())

            if "seg" in wi and wi.seg.parsed:
                np.save(img_out_dir / "seg.npy", wi.seg.parsed.to_numpy())

            if "normal" in wi and wi.normal.parsed:
                np.save(img_out_dir / "normal.npy", wi.normal.parsed.to_numpy())

            if "pose" in wi and wi.pose.parsed:
                # stratum-hq pose.npy is typically an array of (x, y, conf)
                arr = np.array(
                    [[kp.x, kp.y, kp.confidence] for kp in wi.pose.parsed.keypoints],
                    dtype=np.float32
                )
                np.save(img_out_dir / "pose.npy", arr)

            if "dinov3_cls" in wi and wi.dinov3_cls.parsed:
                arr = np.array(wi.dinov3_cls.parsed.embedding, dtype=np.float32)
                np.save(img_out_dir / "dinov3_cls.npy", arr)

            if "dinov3_full" in wi and wi.dinov3_full.parsed:
                arr = np.array(wi.dinov3_full.parsed.patches, dtype=np.float32)
                np.save(img_out_dir / "dinov3_patches.npy", arr)

            if "clip" in wi and wi.clip.parsed:
                arr = np.array(wi.clip.parsed.embedding, dtype=np.float32)
                np.save(img_out_dir / "clip.npy", arr)

            if "t5" in wi and wi.t5.parsed:
                hidden, mask = wi.t5.parsed.to_numpy()
                np.save(img_out_dir / "t5_hidden.npy", hidden)
                np.save(img_out_dir / "t5_mask.npy", mask)

            print(f"  -> Saved stratum-hq compatible outputs to {img_out_dir}/")

        except Exception as e:
            print(f"  -> Failed to process {img_path.name}: {e}")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Process images via Stratum API to stratum-hq format.")
    parser.add_argument("input_folder", help="Path to folder containing images")
    parser.add_argument("output_folder", help="Path to save the extracted .npy and text files")
    args = parser.parse_args()

    api_key = os.environ.get("STRATUM_API_KEY")
    if not api_key:
        print("Please set STRATUM_API_KEY environment variable.")
        sys.exit(1)

    process_local_directory(args.input_folder, args.output_folder, api_key)
