# HavenCore

Self-hosted Smart Home

[Edge Device Code](https://github.com/ThatMattCat/havencore-edge/tree/main)


This project is a work in progress.
This repo doesn't quite contain all necessary info (xtts2 model missing, at minimum) - Updates to come

### Requirements

This project currently has *very* specific requirements but I'll template it out shortly. Works on Ubuntu 24.04 with SuperMicro H12SSL-I server connected to four RTX 3090 GPUs

1. Users must run/have access to an LLM with OpenAI-like API endpoints
2. Graphics cards (currently hard-coded to require four GPUs, will fix)

### Install

```
git clone https://github.com/ThatMattCat/havencore.git
```

### Run

```
cd havencore
docker compose up -d
```
