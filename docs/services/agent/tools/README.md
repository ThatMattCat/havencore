# Agent Tools (MCP Servers)

The agent's tool-calling surface is split across several Model Context Protocol (MCP) servers, each packaged as a Python module under `services/agent/selene_agent/modules/`. Every MCP server runs as a subprocess of the agent, advertises its tools over stdio, and the agent's `mcp_client_manager` wires them into the LLM's function-calling interface via the `UnifiedTool` abstraction.

## MCP servers

| Server | Module | Tools | Doc |
|--------|--------|-------|-----|
| Home Assistant | `mcp_homeassistant_tools` | 20 — REST/WS control, registry, presence, timer/template/history/calendar (read + create), media transport | [home-assistant.md](home-assistant.md) |
| Plex | `mcp_plex_tools` | 5 — library search + cloud-relay playback | [plex.md](plex.md) |
| Music Assistant | `mcp_music_assistant_tools` | 7 — audio search, player enumeration, queue-aware playback / announcement / transport on speakers + Chromecasts | [music-assistant.md](music-assistant.md) |
| General Tools | `mcp_general_tools` | Up to 7 (credential-gated) — weather, Brave, Wolfram, Wikipedia, ComfyUI, Signal messaging, multimodal vision | [general.md](general.md) |
| Qdrant | `mcp_qdrant_tools` | 3 — semantic memory store/search/delete on Qdrant + bge embeddings | [qdrant.md](qdrant.md) |
| MQTT / Cameras | `mcp_mqtt_tools` | 1 — camera snapshot trigger via HA + MQTT round-trip | [mqtt.md](mqtt.md) |
| GitHub | `mcp_github_tools` | 7 — repo code search / read / list / pull-latest + list/get/create GitHub Issues | [github.md](github.md) |
| Face Recognition | `mcp_face_tools` | 5 — who's at a camera, recent visitors, list/enroll/access-level for known people | [face.md](face.md) |
| Vision | `mcp_vision_tools` | 5 — describe image / camera snapshot / compare two images / identify object / read text in image (wraps the `vllm-vision` service) | [vision.md](vision.md) |
| Reminder | `mcp_reminder_tools` | 3 — schedule one-shot or recurring reminders backed by the autonomy engine, list, cancel | [reminder.md](reminder.md) |
| Device Actions | `mcp_device_action_tools` | 5 — `set_alarm` (fire-and-forget intent); `take_photo` / `identify_object_in_photo` / `read_text_from_image` / `who_is_in_view` round-trip a JPEG via `/api/companion/upload`, with the vision-chained variants forwarding the captured `image_url` to the vision pipeline server-side and `who_is_in_view` POSTing the JPEG to face-recognition's `/api/identify` | [device-action.md](device-action.md) |

## Writing new tools

See [development.md](development.md) for the authoring workflow — module layout, tool-definition pattern, configuration, logging, and testing.

## Related docs

- [Agent Service](../README.md) — how tools are orchestrated by the agent
- [Media Control](../../../integrations/media-control.md) — cross-cutting topic spanning the Plex, Music Assistant, and HA tool servers
- [Home Assistant Integration](../../../integrations/home-assistant.md) — end-user HA setup
- [API Reference](../../../api-reference.md) — `/api/tools` and `/mcp/status` endpoints
