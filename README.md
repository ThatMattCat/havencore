# HavenCore

Self-Hosted AI Smart Home

## Overview

This is a personal project slowly being templated for more general use. It is designed to host multiple AI models (audio, text, etc) and act as a voice-activated smart home AI. Current models/functionality:

* Integrates with an [Edge Device with Wake-Word Activation](https://github.com/ThatMattCat/havencore-edge/tree/main) (eg: Replacement for Alexa or Google Home) - Designed for RPi3 with [ReSpeaker 4-Mic Array](https://www.seeedstudio.com/ReSpeaker-USB-Mic-Array-p-4247.html) (testing others soon) and a speaker connected to the Pi.
* Speech-To-Text - Near Realtime Transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
* Text-To-Speech - Converts LLM response to audio using [Coqui xTTSv2](https://github.com/coqui-ai/TTS). Results are played back on the Edge device.
* Integrates with Grafana Loki for logging. This is hard-coded/required right now but will be optional soon. Uses the Loki ingest API, tested working with locally-hosted Loki but _should_ work for Cloud as well, with a few changes to `shared/scripts/logger.py`
  * The logger.py also includes comments to replace Loki with LogScale, this worked in the past but is no longer tested. This will be streamlined with configs at some point.
  * The logs include `trace_id` values with each trace unique to a 'conversation', for easier analysis. Right now this is custom code but should be migrated to OpenTelemetry standards.
* _Large Language Model_ - Not yet included in these containers, must host on your own. Any OpenAI-like API with tool-calling should work but I  use llama.cpp(Qwen3 235B Instruct GGUF) and vLLM (Mistral Large 2411 AWQ).

It is written to use the following as tools (function calling):

* Home Assistant - Uses the Python [HomeAssistant-API](https://pypi.org/project/HomeAssistant-API/) package to get device states and perform actions. WARN: Currently no safeguards here, it can try to use any entity/service. The built-in LLM system prompt works for larger AI models (they have enough built-in knowledge to utilize actions) but some may need further information in the system prompt.
* Brave Search - Used for general web searches to collect URLs. The free API only provides web searches so this doesn't have much use yet without a scraped to parse the search results (TBD). Paid versions of Brave API provide much more value(eg: Summarized results to avoid scraping) and should also be integrated in the future.
* WolframAlpha - Free API so why not. Doesn't get much use on a day-to-day basis but cool when its needed.
* More To Come - These are easy to add, just keeping it minimal during the initial build

This project aims to avoid high-level AI libraries (LangChain,etc) in order to allow a wider variety of lower-level setups. That requires a little more boiler-plate code but allows more flexibility when choosing the local AI models & hosting solutions, especially when working in the limits of a Consumer-hosted system like a homelab.

### Workflow

1. Edge device sends a control message to 'speech-to-text' container to start transcribing
2. Streamed audio is received on WebSocket port 6000 of speech-to-text container
3. Edge device sends control message to stop transcribing (eg: After 2 seconds of silence)
4. The speech-to-text container sends the text transcription to Agent container and receives text response
5. Agent text response is sent to text-to-speech container, which replies with a URL to the audio generated from response
6. Response audio URL is sent to edge device, which plays it through a speaker
7. Tools reset state to prepare for next communication

### Requirements

This project currently has *very* specific requirements but will be templated out shortly. Works on Ubuntu 22.04 with SuperMicro H12SSL-I server connected to four RTX 3090 GPUs

1. Users must run/have access to an LLM with OpenAI-like API endpoints
2. Graphics cards (currently hard-coded to require four GPUs, will fix)

### Install

```
git clone https://github.com/ThatMattCat/havencore.git
```

### Configure

Required configurations are in `shared/configs/shared_config.py`

### Run

```
cd havencore
docker compose up -d
```
