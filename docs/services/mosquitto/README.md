# Mosquitto MQTT Broker

Eclipse Mosquitto MQTT broker. Used by the agent's MCP MQTT server for camera-snapshot round-trips with Home Assistant, and available for any other MQTT-based integration you want to wire up.

## Status

Documentation stub — the broker runs with a default config today. Expand this doc if you add auth, bridging, persistence tuning, or more elaborate topic schemes.

## Ports

- `1883` — MQTT (TCP)
- `9001` — MQTT over WebSocket

## Where to look in the meantime

- `services/mosquitto/` — config and Dockerfile
- `compose.yaml` — runtime configuration and volume mounts
- [MCP MQTT / Cameras](../agent/tools/mqtt.md) — agent-side integration (the only consumer today)
