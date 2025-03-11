import os
import shutil
import subprocess
import tempfile
import hashlib
from pathlib import Path

def get_filename_from_url(url: str, extension: str) -> str:
    """Generate a unique filename from a URL using MD5 hash"""
    filename = hashlib.md5(url.encode()).hexdigest()
    return f"{filename}.{extension}"

def download_replicate_weights(url: str, path: Path):
    """Downloads weights from a Replicate tarball URL and extracts the safetensors file"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        extract_dir = temp_dir / "weights"

        try:
            subprocess.run(["pget", "-x", url, extract_dir], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download tarball: {e}")

        # Find safetensors file
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

        # Generate unique filename based on URL and move the file
        unique_filename = get_filename_from_url(url, "safetensors")
        target_path = path.parent / unique_filename
        shutil.move(safetensors_paths[0], target_path)

    return unique_filename
