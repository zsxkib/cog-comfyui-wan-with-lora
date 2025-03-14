import os
import shutil
import subprocess
import tempfile
import hashlib
from pathlib import Path


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def get_filename_from_url(url: str, model_type: str) -> str:
    return f"{model_type}_{url_hash(url)}.safetensors"


def download_replicate_weights(url: str, lora_dir: str):
    """Downloads weights from a Replicate tarball URL and extracts the safetensors file"""
    hash = url_hash(url)

    # Check if either version already exists
    lora_dir_path = Path(lora_dir)
    existing_14b = lora_dir_path / f"14b_{hash}.safetensors"
    existing_1_3b = lora_dir_path / f"1.3b_{hash}.safetensors"

    if existing_14b.exists():
        print(f"✅ {existing_14b.name} already cached")
        return existing_14b.name, "14b"
    if existing_1_3b.exists():
        print(f"✅ {existing_1_3b.name} already cached")
        return existing_1_3b.name, "1.3b"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        extract_dir = temp_dir / "weights"

        try:
            subprocess.run(["pget", "-x", url, extract_dir], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download tarball: {e}")

        safetensors_paths = []
        for root, _, files in os.walk(extract_dir):
            root = Path(root)
            for filename in files:
                file_path = root / filename
                if file_path.suffix == ".safetensors":
                    safetensors_paths.append(file_path)

        if not safetensors_paths:
            raise ValueError("No .safetensors file found in tarball")
        if len(safetensors_paths) > 1:
            raise ValueError("Multiple .safetensors files found in tarball")

        model_type = "1.3b" if "1.3b" in safetensors_paths[0].name.lower() else "14b"
        unique_filename = get_filename_from_url(url, model_type)
        target_path = Path(lora_dir) / unique_filename
        shutil.move(safetensors_paths[0], target_path)

        return unique_filename, model_type
