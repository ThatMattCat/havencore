# cuDNN 9 in the base — onnxruntime-gpu 1.19+ requires it (1.18 needed
# cuDNN 8, but its CUDA-12 wheel was rebuilt against cuDNN 9, so either
# version pulls us into the cuDNN-9 world). Matching CUDA 12.4 keeps the
# stack on a single CUDA major across services.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# System libs needed by opencv-python-headless and insightface's image I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./

EXPOSE 6006

ENTRYPOINT ["python3"]
CMD ["-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6006"]
