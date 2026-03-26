FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    sox \
    espeak-ng \
    libespeak-ng-dev \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY pyproject.toml MANIFEST.in /workspace/

RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir psutil packaging ninja

RUN pip install --no-cache-dir flash-attn --no-build-isolation && \
	pip install --no-cache-dir -e .
