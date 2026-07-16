FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
WORKDIR /app

# Rhubarb Lip Sync — viseme timeline generator. Used post-Kokoro to
# produce the X-Visemes response header consumed by the companion app's
# Live2D overlay for accurate mouth animation. Not in apt; install from
# the upstream GitHub release.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget unzip ca-certificates \
    && wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/Rhubarb-Lip-Sync-1.14.0-Linux.zip -O /tmp/rhubarb.zip \
    && unzip -q /tmp/rhubarb.zip -d /opt \
    && mv /opt/Rhubarb-Lip-Sync-1.14.0-Linux /opt/rhubarb \
    && ln -s /opt/rhubarb/rhubarb /usr/local/bin/rhubarb \
    && rm /tmp/rhubarb.zip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./
EXPOSE 6005
ENTRYPOINT ["python3"]
CMD ["main.py"]