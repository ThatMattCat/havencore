# MCP Server: MQTT / Camera Snapshots (`mcp_mqtt_tools`)

Reference doc for the MQTT MCP server. Despite the module name, its only
current exposed tool is a camera-snapshot trigger that fires an HA script
over REST and waits for the resulting image URLs on an MQTT topic.

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_mqtt_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_mqtt_tools` |
| Transport | MCP stdio |
| Server name | `havencore-general-tools` (named before the module was split out) |
| MQTT client | `paho-mqtt` with threaded loop |
| Tool count | 1 (conditional on MQTT connectivity) |

The module doubles as a generic MQTT wiring point, but today it ships
with a single `HACamSnapper` integration that coordinates HA and MQTT to
produce camera snapshots. The class is structured so new MQTT-driven
tools can be added alongside it.

## Tool inventory

| Tool | Purpose |
|------|---------|
| `get_camera_snapshots()` | Calls the HA script `script.capture_all_cameras` via REST. That script is expected to capture frames from each camera and publish a JSON payload of URLs to the `home/cameras/snapshots` topic. The tool blocks (up to 10 s) waiting for that MQTT message and returns the URL list. |

The tool is only registered when the MQTT client is connected at
list-tools time — if the broker is unreachable, the tool silently
disappears from the agent's tool surface.

## Required HA side

This tool is not self-contained: it depends on a user-defined Home
Assistant script named `capture_all_cameras` and an MQTT publisher that
posts to the expected topic. The script/publisher are your responsibility
to author — the MCP server only triggers and listens.

Expected message shape on `home/cameras/snapshots`:

```json
{
  "urls": [
    "http://.../camera1.jpg",
    "http://.../camera2.jpg"
  ]
}
```

The server also subscribes to `home/cameras/cleanup/status` (received but
currently not surfaced — a placeholder for a future cleanup workflow).

## Configuration

| Var | Default | Purpose |
|-----|---------|---------|
| `HAOS_URL` | `NO_HAOS_URL_SET` | Used to POST `/services/script/capture_all_cameras`. The trailing `/api` is expected. |
| `HAOS_TOKEN` | `NO_HAOS_TOKEN_SET` | Bearer for the HA REST call. |
| `MQTT_BROKER` | `mosquitto` | Hostname of the MQTT broker (default matches the compose service name). |
| `MQTT_PORT` | `1883` | Broker port. |

Env vars are read directly via `os.getenv()` in `mcp_server.py` — not via
`selene_agent.utils.config`.

The agent spawns the server via `MCP_SERVERS` in `.env`:

```json
{
  "name": "mqtt",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_mqtt_tools"],
  "enabled": true
}
```

## Internals worth knowing

- **MQTT connection is eager.** `HACamSnapper.__init__` calls
  `mqtt_client.connect(...)` synchronously on module import. If the
  broker isn't up yet, the constructor fails and the server won't start
  cleanly. Compose dependencies are the mitigation — the
  `mosquitto` service should start before the agent.
- **`loop_start()` runs in a worker thread.** The paho client uses its
  own thread for MQTT I/O; the MCP server uses asyncio for tool dispatch.
  Cross-thread signaling happens via an `asyncio.Future`
  (`self._snapshot_future`) that the paho callback resolves when a
  matching message arrives.
- **Only one request can be in flight at a time.** The `_snapshot_future`
  is a single attribute on the class, not a queue. Concurrent calls will
  step on each other. In practice the LLM never calls this twice in the
  same turn.
- **10 s timeout is the maximum wait.** If nothing arrives on the topic,
  the tool returns `{"success": false, "error": "Timeout…", "partial_urls": []}`.
  Any URLs received before the timeout are surfaced in `partial_urls`.
- **Non-200 from HA ≠ tool failure detection is limited.** The tool
  checks the HTTP status of the script trigger but not whether the
  script actually captured anything. A silent failure in the script
  shows up as a timeout on the MQTT side.

## Usage pattern

Because snapshots come back as URLs rather than in-line image data, the
system prompt tells the LLM to chain `get_camera_snapshots` →
`query_multimodal_api` (from `mcp_general_tools`) on each URL to actually
describe what the cameras see. Expect two tool calls per "what's on the
cameras?" query.

## Troubleshooting

### Tool is missing from the tool list

`self.snapshotter.mqtt_client.is_connected()` returned false at
`list_tools()` time. Either:

- Mosquitto isn't running — `docker compose ps mosquitto`.
- The broker hostname / port are wrong — check `MQTT_BROKER` /
  `MQTT_PORT`.
- The agent container can't reach the broker — check Docker networks.

A module restart is needed after the broker recovers: the MQTT client
doesn't reconnect on its own in this implementation.

```bash
docker compose restart agent
```

### Timeout with empty `partial_urls`

- The HA script `capture_all_cameras` exists but didn't publish to
  `home/cameras/snapshots`. Verify the script's final step.
- The MQTT publisher is using a different topic. Topic match is exact.
- QoS mismatch prevented delivery. The tool subscribes with default QoS 0
  — publish with QoS 0 or 1.

### HA REST returns 401 / 403

`HAOS_TOKEN` is wrong or expired. Same token as the Home Assistant MCP
server — regenerate a long-lived access token in HA and restart.

### HA REST returns 404

The script `capture_all_cameras` doesn't exist on this HA instance. The
MCP server doesn't create the script — author it yourself (via HA UI or
`scripts.yaml`).

## Related files

- `services/agent/selene_agent/modules/mcp_mqtt_tools/mcp_server.py` —
  implementation.
- Mosquitto service in `compose.yaml` — broker.

## See also

- [MCP General](general.md) — the multimodal analysis tool typically
  chained after this one.
- [Home Assistant Integration](../../../integrations/home-assistant.md) — where the
  `capture_all_cameras` script is authored.
