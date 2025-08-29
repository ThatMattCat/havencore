FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get update && apt-get install -y curl nodejs
COPY pyproject.toml ./
COPY selene_agent ./selene_agent/
RUN pip install -e .
EXPOSE 6002
CMD ["agent"]