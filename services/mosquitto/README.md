# mosquitto

MQTT broker. Used by the autonomy engine's reactive subscriber to
listen on live topic feeds (HA state changes, external sensors) and
by `mcp_mqtt_tools` for agent-initiated publish/subscribe.

| | |
|---|---|
| **Ports** | `1883` (MQTT), `9001` (WebSocket) |
| **Health** | Healthcheck intentionally disabled in `compose.yaml`. `mosquitto_sub -t '$SYS/#' -C 1` works but trips on config edits during live runs — re-enable if you prefer fail-fast behavior. |
| **Image** | `eclipse-mosquitto:latest` (rollback digest pinned in compose.yaml) |

## Config

- `config/mosquitto.conf` — broker config (listeners, auth, logging)
- `data/` — persistent state (retained messages, subscriptions)
- `log/` — broker logs

## Key env

- `AUTONOMY_MQTT_ENABLED=true` — opt in to the agent's reactive MQTT
  subscriber
- `AUTONOMY_MQTT_CLIENT_ID`,
  `AUTONOMY_MQTT_RECONNECT_MAX_SEC` — client tuning

## More

- Deep dive: [../../docs/services/mosquitto/README.md](../../docs/services/mosquitto/README.md)
- MQTT MCP tools:
  `services/agent/selene_agent/modules/mcp_mqtt_tools/`
- Reactive autonomy design:
  [../../docs/services/agent/autonomy/README.md](../../docs/services/agent/autonomy/README.md)
