#!/usr/bin/env bash
set -euo pipefail

echo "🎤 CueCatcher Voice Cloning Setup"
echo "═══════════════════════════════════════════"
echo ""
echo "This sets up fully local voice cloning using the community"
echo "voxtral-voice-clone project (CC BY-NC 4.0 license)."
echo ""
echo "It trains the missing codec encoder that Mistral didn't release"
echo "with the Voxtral-4B-TTS open weights. Once trained, you can"
echo "clone any voice from a 5-25 second audio clip — fully offline."
echo ""

MODEL_DIR="${MODEL_DIR:-./models}"
CLONE_DIR="${MODEL_DIR}/voxtral-voice-clone"

# ── Option 1: Download pre-trained encoder (if available) ──
echo "Checking for pre-trained community encoder..."
if [ -f "${CLONE_DIR}/encoder.pt" ]; then
    echo "✅ Encoder already downloaded: ${CLONE_DIR}/encoder.pt"
    echo ""
    echo "Done! Voice cloning is ready."
    echo "Upload a parent voice clip via the tablet Settings → Voice."
    exit 0
fi

# ── Option 2: Clone the repo and train ──
echo "Pre-trained encoder not found. Setting up from source..."
echo ""

mkdir -p "${CLONE_DIR}"

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ git not found. Install git first."
    exit 1
fi

# Clone the community repo
REPO_DIR="/tmp/voxtral-voice-clone"
if [ -d "${REPO_DIR}" ]; then
    echo "Updating existing clone..."
    cd "${REPO_DIR}" && git pull
else
    echo "Cloning voxtral-voice-clone..."
    git clone https://github.com/Al0olo/voxtral-voice-clone.git "${REPO_DIR}"
fi

cd "${REPO_DIR}"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt
pip install speechbrain --break-system-packages 2>/dev/null || pip install speechbrain

echo ""
echo "═══════════════════════════════════════════"
echo "Repository cloned to: ${REPO_DIR}"
echo ""
echo "To train the encoder, you need a speech dataset."
echo "The project supports LibriSpeech (English) out of the box."
echo ""
echo "Quick training (recommended for first setup):"
echo "  cd ${REPO_DIR}"
echo "  python train_encoder.py --dataset librispeech --split dev-clean --epochs 50"
echo ""
echo "Full training (better quality):"
echo "  cd ${REPO_DIR}"
echo "  python train_encoder.py --dataset librispeech --split train-clean-100 --epochs 200"
echo ""
echo "After training, copy the encoder weights:"
echo "  cp ${REPO_DIR}/checkpoints/best_encoder.pt ${CLONE_DIR}/encoder.pt"
echo ""
echo "Then restart CueCatcher and upload a parent voice clip."
echo ""
echo "═══════════════════════════════════════════"
echo ""
echo "ALTERNATIVE: Use Mistral API instead (easier, no training):"
echo "  export MISTRAL_API_KEY=your_key_here"
echo "  # Then restart CueCatcher — it will auto-detect the API key"
echo ""
echo "Get an API key at: https://console.mistral.ai/"
echo "═══════════════════════════════════════════"
