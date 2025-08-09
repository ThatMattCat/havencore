FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./
EXPOSE 6004
EXPOSE 6003
RUN apt-get update && apt-get install -y curl
ENTRYPOINT ["python3"]
CMD ["main.py"]