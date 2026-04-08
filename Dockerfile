FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    ffmpeg libsndfile1 libgl1-mesa-glx libglib2.0-0 \
    curl git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server/ /app/server/
COPY inference/ /app/inference/
COPY voice/ /app/voice/
COPY config/ /app/config/

EXPOSE 8084 8443

CMD ["python", "-m", "uvicorn", "server.main:app", \
     "--host", "127.0.0.1", "--port", "8084", \
     "--ws-max-size", "16777216", \
     "--timeout-keep-alive", "120"]
