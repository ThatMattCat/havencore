FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
COPY app/requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 6002
CMD ["python", "selene_agent.py"]