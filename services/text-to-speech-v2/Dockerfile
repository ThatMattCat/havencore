# Chatterbox-TTS 0.1.7 pins torch==2.6.0; matching the base avoids a
# torch/torchvision ABI mismatch (which manifests as
# `RuntimeError: operator torchvision::nms does not exist` during
# transformers' import chain).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
WORKDIR /app

# Rhubarb Lip Sync — viseme timeline generator. Same install as v1 so the
# X-Visemes header the companion app already consumes stays byte-for-byte
# identical. See services/text-to-speech-v2/app/rhubarb.py.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget unzip ca-certificates curl \
    && wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.14.0/Rhubarb-Lip-Sync-1.14.0-Linux.zip -O /tmp/rhubarb.zip \
    && unzip -q /tmp/rhubarb.zip -d /opt \
    && mv /opt/Rhubarb-Lip-Sync-1.14.0-Linux /opt/rhubarb \
    && ln -s /opt/rhubarb/rhubarb /usr/local/bin/rhubarb \
    && rm /tmp/rhubarb.zip \
    && rm -rf /var/lib/apt/lists/*

# Bundle reference voice clips from devnen/Chatterbox-TTS-Server (MIT).
# Chatterbox is zero-shot — to get a stable named "voice" the model needs a
# short reference clip per generation. We fetch a set at build time so the
# container has working "preset" voices out of the box. Operators can drop
# their own .wav files into /app/voices via the compose volume mount to add
# or override voices (including a custom Selene clone later).
#
# Stored under /opt so the /app volume mount (live Python source) doesn't
# shadow the bundled set.
RUN mkdir -p /opt/chatterbox-voices \
    && for name in Abigail Alice Cora Elena Emily Gianna Jade Layla Olivia Taylor Adrian Alexander Austin Henry Ian Michael Miles Ryan Thomas Connor ; do \
        curl -fsSL -o "/opt/chatterbox-voices/${name}.wav" \
            "https://raw.githubusercontent.com/devnen/Chatterbox-TTS-Server/main/voices/${name}.wav" \
        || echo "warn: failed to fetch ${name}.wav" ; \
    done

COPY NOTICE ./NOTICE
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./

EXPOSE 6015
ENTRYPOINT ["python3"]
CMD ["main.py"]
