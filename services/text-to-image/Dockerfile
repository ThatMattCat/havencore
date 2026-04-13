FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone ComfyUI at a pinned commit so fresh builds stay reproducible.
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app \
    && git -C /app checkout 31283d2892f54caf9bfdf6edb9c98cbfa88c5f0c

# Install Python dependencies. Torch is pinned to the last trio verified
# working on NVIDIA driver 580.x; newer torch has required newer drivers.
RUN pip3 install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    && pip3 install --no-cache-dir -r requirements.txt

# Install additional useful packages for API usage
RUN pip3 install --no-cache-dir \
    aiofiles \
    python-multipart

# # Create necessary directories
# RUN mkdir -p \
#     /app/models/checkpoints \
#     /app/models/clip \
#     /app/models/clip_vision \
#     /app/models/controlnet \
#     /app/models/embeddings \
#     /app/models/loras \
#     /app/models/upscale_models \
#     /app/models/vae \
#     /app/output \
#     /app/input \
#     /app/custom_nodes

# Expose port for web UI
EXPOSE 8188

# Set environment variable for better memory management
ENV MALLOC_MMAP_THRESHOLD_=131072

# Start ComfyUI with API enabled
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--default-device", "3"]