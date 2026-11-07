# ==============================================================================
# MEPhI AI Photonics Hackathon - DOE Phase Reconstruction
# Multi-platform Docker Container (CPU & NVIDIA GPU support)
# ==============================================================================

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Metadata
LABEL maintainer="AI Photonics Team" \
      description="Diffractive Optical Element (DOE) phase reconstruction via Deep U-Net" \
      version="1.0"

# Configure environment variables:
# - PYTHONUNBUFFERED=1 ensures realtime logging without stdout buffering
# - PYTHONDONTWRITEBYTECODE=1 prevents creating .pyc clutter in container
# - DEBIAN_FRONTEND=noninteractive suppresses interactive apt prompts
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install minimal system packages for SSL certs and headless graphics/runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specifications first for optimal Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, weights, and documentation
COPY src/ /app/src/
COPY doe_network.pth /app/doe_network.pth
COPY baseline_2layer_network.pth /app/baseline_2layer_network.pth
COPY README.md /app/README.md

# Set entrypoint to run the inference CLI script
ENTRYPOINT ["python", "src/infer.py"]

# Default command displays usage help if no parameters are supplied
CMD ["--help"]
