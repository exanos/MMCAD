#!/usr/bin/env python3
"""
Download AutoBrep pretrained weights from HuggingFace.

Downloads the surface and edge FSQ VAE checkpoints required for
initializing the B-Rep encoder in CLIP4CAD-H.

Usage:
    python scripts/download_autobrep_weights.py
    python scripts/download_autobrep_weights.py --output-dir pretrained/autobrep

HuggingFace repository: https://huggingface.co/SamGiantEagle/AutoBrep
"""

import argparse
import os
from pathlib import Path


def download_weights(output_dir: str = "pretrained/autobrep", force: bool = False):
    """
    Download AutoBrep weights from HuggingFace.

    Args:
        output_dir: Directory to save weights
        force: Force re-download even if files exist
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        return False

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "SamGiantEagle/AutoBrep"

    print(f"Downloading AutoBrep weights from {repo_id}")
    print(f"Output directory: {output_path.absolute()}")
    print()

    try:
        # List available files
        print("Fetching file list from HuggingFace...")
        files = list_repo_files(repo_id)
        print(f"Found {len(files)} files:")
        for f in files:
            print(f"  - {f}")
        print()

        # Download checkpoint files
        downloaded = []
        for filename in files:
            # Download .pt, .pth, .ckpt, .bin files
            if any(filename.endswith(ext) for ext in [".pt", ".pth", ".ckpt", ".bin", ".safetensors"]):
                local_path = output_path / filename

                if local_path.exists() and not force:
                    print(f"Skipping {filename} (already exists)")
                    downloaded.append(str(local_path))
                    continue

                print(f"Downloading {filename}...")
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False,
                )
                downloaded.append(path)
                print(f"  Saved to: {path}")

        # Also download config files if present
        for filename in files:
            if any(filename.endswith(ext) for ext in [".yaml", ".json", ".yml"]):
                local_path = output_path / filename

                if local_path.exists() and not force:
                    print(f"Skipping {filename} (already exists)")
                    continue

                print(f"Downloading {filename}...")
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(output_path),
                    local_dir_use_symlinks=False,
                )
                print(f"  Saved to: {path}")

        print()
        print("=" * 60)
        print("Download complete!")
        print("=" * 60)
        print()
        print("Downloaded checkpoint files:")
        for p in downloaded:
            print(f"  - {p}")

        print()
        print("To use these weights in CLIP4CAD-H, update your config:")
        print()
        print("  model:")
        print("    brep:")
        print(f"      surface_checkpoint: {output_path}/surface_fsq_vae.pt")
        print(f"      edge_checkpoint: {output_path}/edge_fsq_vae.pt")
        print()
        print("Or set the AUTOBREP_WEIGHTS environment variable:")
        print(f"  export AUTOBREP_WEIGHTS={output_path.absolute()}")

        return True

    except Exception as e:
        print(f"Error downloading weights: {e}")
        print()
        print("If the repository is private or requires authentication,")
        print("login with: huggingface-cli login")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download AutoBrep pretrained weights from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pretrained/autobrep",
        help="Directory to save weights (default: pretrained/autobrep)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )

    args = parser.parse_args()

    success = download_weights(args.output_dir, args.force)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
