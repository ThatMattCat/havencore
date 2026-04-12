# Agent Tools (MCP Servers)

The agent's tool-calling surface is split across several Model Context Protocol (MCP) servers, each packaged as a Python module under `services/agent/selene_agent/modules/`. Every MCP server runs as a subprocess of the agent, advertises its tools over stdio, and the agent's `mcp_client_manager` wires them into the LLM's function-calling interface via the `UnifiedTool` abstraction.

## MCP servers

| Server | Module | Tools | Doc |
|--------|--------|-------|-----|
| Home Assistant | `mcp_homeassistant_tools` | 21 — REST/WS control, registry, presence, timer/template/history/calendar, media transport | [home-assistant.md](home-assistant.md) |
| Plex | `mcp_plex_tools` | 5 — library search + cloud-relay playback | [plex.md](plex.md) |
| General Tools | `mcp_general_tools` | Up to 7 (credential-gated) — weather, Brave, Wolfram, Wikipedia, ComfyUI, email, multimodal vision | [general.md](general.md) |
| Qdrant | `mcp_qdrant_tools` | 2 — semantic memory store/search on Qdrant + bge embeddings | [qdrant.md](qdrant.md) |
| MQTT / Cameras | `mcp_mqtt_tools` | 1 — camera snapshot trigger via HA + MQTT round-trip | [mqtt.md](mqtt.md) |

## Writing new tools

See [development.md](development.md) for the authoring workflow — module layout, tool-definition pattern, configuration, logging, and testing.

## Related docs

- [Agent Service](../README.md) — how tools are orchestrated by the agent
- [Media Control](../../../integrations/media-control.md) — cross-cutting topic spanning the Plex and HA tool servers
- [Home Assistant Integration](../../../integrations/home-assistant.md) — end-user HA setup
- [API Reference](../../../api-reference.md) — `/api/tools` and `/mcp/status` endpoints
