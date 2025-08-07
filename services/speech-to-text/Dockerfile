FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

WORKDIR /app

ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ls -la /usr/local/cuda/lib64/libcudnn* || echo "cuDNN libs not in cuda/lib64"
RUN find /usr -name "libcudnn_ops.so.9*" 2>/dev/null || echo "Looking for cuDNN 9 libs..."
RUN pip3 install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY app .
EXPOSE 5999
EXPOSE 6000
CMD ["python3", "main.py"]