FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./
EXPOSE 6004
EXPOSE 6003
#RUN apt update && apt
ENTRYPOINT ["python3"]
CMD ["main.py"]