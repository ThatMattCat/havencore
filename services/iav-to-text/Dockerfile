FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install supervisor via pip to get a Python 3.12+ compatible version
RUN pip install supervisor

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies including vllm[audio]
RUN pip install --no-cache-dir -r requirements.txt

# Create uploads directory
RUN mkdir -p /app/uploads

# Copy the Gradio application
COPY app/gradio_app.py /app/

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set Python to unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Expose both ports (vLLM and Gradio)
EXPOSE 8100 8110

# Override the ENTRYPOINT from vLLM base image
ENTRYPOINT []

# Use supervisor to run both services
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]