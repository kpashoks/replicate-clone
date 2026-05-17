"""
One-time model bootstrap for the wan-animate server.

The main app's README already documents downloading Wan 2.2 Animate weights
and SAM2 to the Network Volume (see step 6 of the bootstrap). This script
is a convenience wrapper for the SAM2 weights specifically — the file that
M3.5's bootstrap didn't include.

Run from a temp Pod with the Network Volume mounted at /workspace, NOT from
inside the wan-animate container at runtime. The container is meant to read
weights from the volume that's already populated.

Usage:
    pip install huggingface_hub
    export VOLUME_ROOT=/workspace/ComfyUI
    python setup_model.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = [
    # (subdir under VOLUME_ROOT/models, filename on disk, HF repo, file in repo)
    (
        "sam2",
        "sam2_hiera_base_plus.safetensors",
        "Kijai/sam2-safetensors",
        "sam2_hiera_base_plus.safetensors",
    ),
    # Wan 2.2 Animate 14B should already be downloaded by the main app's
    # bootstrap. This is here for completeness; comment out if you don't want
    # to re-verify.
    (
        "diffusion_models",
        "wan2.2_animate_14B_bf16.safetensors",
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors",
    ),
]


def _hf_download(repo: str, file_in_repo: str, dest_dir: Path) -> None:
    """Wrap `hf download` for clearer error messages."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] hf download {repo} {file_in_repo} -> {dest_dir}")
    proc = subprocess.run(
        [
            "hf", "download", repo, file_in_repo,
            "--local-dir", str(dest_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"[setup] FAILED: stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}", file=sys.stderr)
        sys.exit(proc.returncode)


def main() -> None:
    volume_root = Path(os.environ.get("VOLUME_ROOT", "/workspace/ComfyUI"))
    if not volume_root.exists():
        print(f"[setup] VOLUME_ROOT={volume_root} does not exist. "
              f"Is the Network Volume mounted? "
              f"On a temp RunPod Pod with the volume attached, set "
              f"VOLUME_ROOT=/workspace/ComfyUI.", file=sys.stderr)
        sys.exit(1)

    for subdir, filename, repo, file_in_repo in REQUIRED_FILES:
        target = volume_root / "models" / subdir / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"[setup] OK: {target} (already present)")
            continue

        # Some HF files are nested inside repo subfolders (e.g. "split_files/...");
        # hf download preserves that path under --local-dir, so we may need
        # to move the file up afterward.
        download_root = volume_root / "models" / subdir / "_hf_tmp"
        _hf_download(repo, file_in_repo, download_root)
        downloaded = download_root / file_in_repo
        if not downloaded.exists():
            print(f"[setup] FAILED: expected {downloaded} after download", file=sys.stderr)
            sys.exit(1)
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded.rename(target)
        shutil.rmtree(download_root, ignore_errors=True)
        print(f"[setup] OK: {target}")

    print("[setup] All required files present.")


if __name__ == "__main__":
    main()
