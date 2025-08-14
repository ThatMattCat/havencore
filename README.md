# HavenCore

Self-Hosted AI Smart Home

### This is a work-in-progress. It works for the creator but requires some manipulation in other environments

## Overview

This is a personal project slowly being templated for more general use. It is designed to host multiple AI models (audio, text, etc) and act as a voice-activated smart home AI. Current models/functionality:

* **Integrates** with an [Edge Device with Wake-Word Activation](https://github.com/ThatMattCat/havencore-edge/tree/main) (eg: Replacement for Alexa or Google Home)
  * New nginx-fronted endpoint on port 80 allows integration with ESP32-Box-3 and similar tools that utilize OpenAI-like
* **Speech-To-Text** - Near Realtime Transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
* **Text-To-Speech** - Converts LLM response to audio using [Coqui xTTSv2](https://github.com/coqui-ai/TTS). Results are played back on the Edge device.
* **Large Language Model** - Currently uses vLLM, although any OpenAI-like API with tool-calling should work.
* **Nginx** - Provides a unified endpoint for OpenAI-like API requests for basic transcription, speech, and chat completion. Routes requests to appropriate container services.
  * /v1/chat/completions - Agent Service text-to-text
  * /v1/audio/speech - text-to-speech Service for audio generation
  * /v1/audio/transcriptions - speech-to-text for audio transcription

It is written to use the following as tools (function calling):

* **Home Assistant** - Uses the Python [HomeAssistant-API](https://pypi.org/project/HomeAssistant-API/) package to get device states and perform actions. WARN: Currently no safeguards here, it can try to use any entity/service.
* **Brave Search** - Used for general web searches to collect URLs. The free API only provides web searches so this doesn't have much use yet without a scraped to parse the search results (TBD). Paid versions of Brave API provide much more value(eg: Summarized results to avoid scraping) and should also be integrated in the future.
* **WolframAlpha** - Free API so why not. Doesn't get much use on a day-to-day basis but cool when its needed.
* **More To Come** - These are easy to add, just keeping it minimal during the initial build

This project aims to avoid high-level AI libraries (LangChain,etc) in order to allow a wider variety of lower-level setups. That requires a little more boiler-plate code but allows more flexibility when choosing the local AI models & hosting solutions, especially when working in the limits of a Consumer-hosted system like a homelab.

### Workflow

#### With RPi3 Edge Device 
1. Edge device sends a control message to 'speech-to-text' container to start transcribing
2. Streamed audio is received on WebSocket port 6000 of speech-to-text container
3. Edge device sends control message to stop transcribing (eg: After 2 seconds of silence)
4. The speech-to-text container sends the text transcription to Agent container and receives text response
5. Agent text response is sent to text-to-speech container, which replies with a URL to the audio generated from response
6. Response audio URL is sent to edge device, which plays it through a speaker
7. Tools reset state to prepare for next communication

#### With ESP32-Box-3
1. Flash ESP32-Box-3 with espressif's provided ChatGPT code
2. Configure the endpoint to be HOST_IP:80/v1/ and set key to match your configuration
3. Enjoy! :)

### Requirements

This project currently has *very* specific requirements but will be templated out shortly. Works on Ubuntu 22.04 with SuperMicro H12SSL-I server connected to four RTX 3090 GPUs

There are some configurations that haven't been documented, mainly in the compose.yaml, .env, and shared_config.py, which control how the services use GPUs (which GPU they access), the LLM model being loaded, and similar. These would need to be modified for most environments. On top of that, the `text-to-speech` service would be missing some models in `text-to-speech/app/models` (xTTSv2, potentially one or two others). This will all be corrected in the future.

The tool is hard-coded/required to use Grafana Loki for logging right now but that will be optional soon. Uses the Loki ingest API, tested working with locally-hosted Loki but _should_ work for Cloud as well, with a few changes to `shared/scripts/logger.py`
  * The logger.py also includes comments to replace Loki with LogScale, this worked in the past but is no longer tested. This will be streamlined with configs at some point.
  * The logs include `trace_id` values with each trace unique to a 'conversation', for easier analysis. Right now this is custom code but should be migrated to OpenTelemetry standards.

### Install

```
git clone https://github.com/ThatMattCat/havencore.git
```

### Configure

Rename `shared/configs/shared_config.py.tmpl` to `shared/configs/shared_config.py` and change all values to fit your environment. 

### Run

```
cd havencore
docker compose up -d
```
