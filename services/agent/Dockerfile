FROM node:18-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY selene_agent ./selene_agent/
RUN pip install -e .
COPY --from=frontend-build /frontend/build /app/static
EXPOSE 6002
CMD ["agent"]
