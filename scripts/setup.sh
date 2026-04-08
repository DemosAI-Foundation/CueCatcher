#!/usr/bin/env bash
set -euo pipefail

echo "🧭 CueCatcher Setup"
echo "═══════════════════════════════════════════"

# ── Check prerequisites ──
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install: https://docs.docker.com/engine/install/"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Install NVIDIA drivers."
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "⚠️  NVIDIA Container Toolkit may not be installed."
    echo "   Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Generate self-signed certs (needed for camera access on mobile) ──
echo "Generating self-signed TLS certificates..."
mkdir -p certs
if [ ! -f certs/cert.pem ]; then
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout certs/key.pem -out certs/cert.pem \
        -days 365 -subj "/CN=CueCatcher.local" \
        -addext "subjectAltName=DNS:CueCatcher.local,IP:$(hostname -I | awk '{print $1}')" \
        2>/dev/null
    echo "  ✅ Certificates generated (certs/cert.pem, certs/key.pem)"
    echo "  ⚠️  You'll need to accept the self-signed cert on your tablet"
else
    echo "  ✅ Certificates already exist"
fi

# ── Create data directories ──
echo "Creating data directories..."
mkdir -p data/{sessions,calibration,voice}
echo "  ✅ Data directories created"

# ── Build and start Docker stack ──
echo ""
echo "Building Docker images..."
docker compose build

echo ""
echo "Starting services..."
docker compose up -d

echo ""
echo "Waiting for services..."
sleep 5

# ── Health check ──
if curl -sf http://localhost:8084/health > /dev/null 2>&1; then
    echo "✅ CueCatcher server is running!"
else
    echo "⚠️  Server may still be starting. Check: docker compose logs -f CueCatcher-server"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "🧭 CueCatcher is ready!"
echo ""
echo "  Server:  http://$(hostname -I | awk '{print $1}'):8084"
echo "  API docs: http://$(hostname -I | awk '{print $1}'):8084/docs"
echo ""
echo "  Next steps:"
echo "  1. Open the URL above on your tablet"
echo "  2. Allow camera + microphone access"
echo "  3. Upload a voice clip (Settings → Voice)"
echo "  4. Start a session"
echo ""
echo "  To stop:  docker compose down"
echo "  To logs:  docker compose logs -f"
echo "═══════════════════════════════════════════"
