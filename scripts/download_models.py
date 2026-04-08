#!/usr/bin/env python3
"""
Download all COMPASS ML models to the local model cache.

Models are downloaded from HuggingFace and torch hub.
Total download: ~4-5 GB
Total VRAM after loading: ~11-12 GB (leaves room for Voxtral TTS)
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))

MODELS = {
    # ── Pose Estimation ──
    "rtmpose": {
        "desc": "RTMPose-l (133 whole-body keypoints)",
        "files": [
            {
                "url": "https://huggingface.co/open-mmlab/mmpose/resolve/main/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd42_20230728.pth",
                "dest": "pose/rtmpose-l.pth",
                "size_mb": 250,
            },
        ],
        "phase": 2,
    },
    "yolo11_pose": {
        "desc": "YOLO11n-pose (person detection + 17 keypoints)",
        "files": [
            {
                "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
                "dest": "pose/yolo11n-pose.pt",
                "size_mb": 12,
            },
        ],
        "phase": 2,
    },

    # ── Gaze Estimation ──
    "l2cs_net": {
        "desc": "L2CS-Net (fine-grained gaze estimation)",
        "files": [
            {
                "url": "https://huggingface.co/Ahmednull/l2cs-net/resolve/main/L2CSNet_gaze360.pkl",
                "dest": "gaze/l2cs_gaze360.pkl",
                "size_mb": 90,
            },
        ],
        "phase": 2,
    },
    "6drepnet": {
        "desc": "6DRepNet (6DOF head pose estimation)",
        "files": [
            {
                "url": "https://huggingface.co/osanseviero/6DRepNet_300W_LP_AFLW2000/resolve/main/model.pth",
                "dest": "gaze/6drepnet.pth",
                "size_mb": 85,
            },
        ],
        "phase": 2,
    },

    # ── Face Analysis ──
    "libreface": {
        "desc": "LibreFace (12 AUs + 7 expressions, ResNet-18)",
        "files": [
            {
                "url": "https://huggingface.co/Boese0601/LibreFace/resolve/main/libreface_au_swin_base.pth",
                "dest": "face/libreface_au.pth",
                "size_mb": 330,
            },
            {
                "url": "https://huggingface.co/Boese0601/LibreFace/resolve/main/libreface_fer_swin_base.pth",
                "dest": "face/libreface_fer.pth",
                "size_mb": 330,
            },
        ],
        "phase": 2,
    },

    # ── Audio Classification ──
    "panns_cnn14": {
        "desc": "PANNs CNN14 (audio event detection, AudioSet-trained)",
        "files": [
            {
                "url": "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth",
                "dest": "audio/panns_cnn14.pth",
                "size_mb": 320,
            },
        ],
        "phase": 2,
    },

    # ── Action Recognition ──
    "poseconv3d": {
        "desc": "PoseConv3D (skeleton-based action recognition)",
        "files": [
            {
                "url": "https://huggingface.co/open-mmlab/mmaction2/resolve/main/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.pth",
                "dest": "action/poseconv3d_ntu60.pth",
                "size_mb": 130,
            },
        ],
        "phase": 2,
    },

    # ── Voxtral TTS ──
    "voxtral_tts": {
        "desc": "Voxtral-4B-TTS (text-to-speech with voice cloning)",
        "files": [],  # Downloaded via HuggingFace hub / vLLM auto-download
        "hf_repo": "mistralai/Voxtral-4B-TTS-2603",
        "phase": 1,
        "note": "Downloaded automatically by vLLM on first load (~8 GB BF16, ~3 GB quantized)",
    },
}


def download_file(url: str, dest: Path, expected_mb: int = 0):
    """Download a file with progress bar."""
    import urllib.request
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        actual_mb = dest.stat().st_size / (1024 * 1024)
        if expected_mb > 0 and actual_mb > expected_mb * 0.9:
            logger.info(f"  ✅ Already exists: {dest.name} ({actual_mb:.0f} MB)")
            return True
        else:
            logger.warning(f"  ⚠️  Incomplete download, re-downloading: {dest.name}")

    logger.info(f"  ⬇️  Downloading: {dest.name} ({expected_mb} MB)...")

    try:
        tmp = dest.with_suffix(".tmp")
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    pct = downloaded / total * 100
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    print(f"\r    {bar} {pct:.0f}%  ({downloaded / 1024 / 1024:.0f}/{total / 1024 / 1024:.0f} MB)", end="", flush=True)

            print()  # newline after progress bar

        shutil.move(str(tmp), str(dest))
        logger.info(f"  ✅ Downloaded: {dest.name}")
        return True

    except Exception as e:
        logger.error(f"  ❌ Failed: {dest.name} — {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def download_hf_repo(repo_id: str, dest: Path):
    """Download a HuggingFace model repo."""
    try:
        from huggingface_hub import snapshot_download

        logger.info(f"  ⬇️  Downloading HuggingFace repo: {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        logger.info(f"  ✅ Downloaded: {repo_id}")
        return True
    except ImportError:
        logger.error("  ❌ huggingface_hub not installed: pip install huggingface-hub")
        return False
    except Exception as e:
        logger.error(f"  ❌ Failed: {repo_id} — {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download COMPASS ML models")
    parser.add_argument("--phase", type=int, default=None, help="Only download models for this phase (1 or 2)")
    parser.add_argument("--model", type=str, default=None, help="Only download a specific model")
    parser.add_argument("--dir", type=str, default=str(MODEL_DIR), help="Model directory")
    parser.add_argument("--list", action="store_true", help="List all models without downloading")
    args = parser.parse_args()

    model_dir = Path(args.dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("\n🧭 COMPASS ML Models\n")
        for name, info in MODELS.items():
            total_mb = sum(f.get("size_mb", 0) for f in info.get("files", []))
            phase = info.get("phase", "?")
            note = info.get("note", "")
            print(f"  [{phase}] {name:20s}  {info['desc']}")
            if total_mb:
                print(f"      Download: {total_mb} MB")
            if note:
                print(f"      Note: {note}")
            print()
        return

    logger.info(f"🧭 COMPASS Model Downloader")
    logger.info(f"   Model directory: {model_dir}")
    print()

    models_to_download = MODELS
    if args.model:
        if args.model not in MODELS:
            logger.error(f"Unknown model: {args.model}. Use --list to see available models.")
            sys.exit(1)
        models_to_download = {args.model: MODELS[args.model]}
    elif args.phase:
        models_to_download = {k: v for k, v in MODELS.items() if v.get("phase") == args.phase}

    total_files = sum(len(m.get("files", [])) for m in models_to_download.values())
    hf_repos = sum(1 for m in models_to_download.values() if m.get("hf_repo"))
    logger.info(f"Downloading {total_files} files + {hf_repos} HF repos\n")

    success = 0
    failed = 0

    for name, info in models_to_download.items():
        logger.info(f"📦 {name}: {info['desc']}")

        for file_info in info.get("files", []):
            dest = model_dir / file_info["dest"]
            ok = download_file(file_info["url"], dest, file_info.get("size_mb", 0))
            if ok:
                success += 1
            else:
                failed += 1

        if info.get("hf_repo"):
            dest = model_dir / name
            ok = download_hf_repo(info["hf_repo"], dest)
            if ok:
                success += 1
            else:
                failed += 1

        if info.get("note"):
            logger.info(f"  ℹ️  {info['note']}")

        print()

    print()
    logger.info(f"Done! {success} succeeded, {failed} failed")
    logger.info(f"Models stored in: {model_dir}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
