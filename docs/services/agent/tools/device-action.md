# MCP Server: Device Actions (`mcp_device_action_tools`)

Reference doc for the device-side action MCP server. These tools do
not perform their action server-side — they exist so the LLM has a
discoverable capability to call, and so the orchestrator has a
`tool_call` to attach a `device_action` event to. The companion app
on the session's device receives that event over `/ws/chat` and fires
the matching platform intent (e.g.
[`AlarmClock.ACTION_SET_ALARM`](https://developer.android.com/reference/android/provider/AlarmClock#ACTION_SET_ALARM)).

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_device_action_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_device_action_tools` |
| Transport | MCP stdio |
| Server name | `havencore-device-action-tools` |
| Backend | None — handlers return a structured status string only. |
| Tool count | 1 |
| Wire-protocol allowlist | `selene_agent.orchestrator.DEVICE_ACTION_TOOLS` (a `frozenset`) |

The execution model is two-sided:

1. **Server side (this module + orchestrator).** The LLM calls
   `set_alarm`. The MCP handler validates the args and returns a
   `{"status": "scheduled", ...}` JSON blob — purely so the LLM has
   something to read back and phrase a natural reply ("Alarm set for
   7 AM."). No DB writes, no external calls.
2. **Wire side (orchestrator → `/ws/chat` → companion app).** When
   the tool name is in `DEVICE_ACTION_TOOLS`, the orchestrator emits
   an extra [`device_action` event](#wire-protocol) right after the
   normal `tool_result`. The companion app's `DeviceActionDispatcher`
   parses it and fires the platform intent on the user's device.

The `tool_call` / `tool_result` pair stays — the chat screen still
renders a `ToolCallCard` server-side breadcrumb. The new event is
what actually fires the intent on the device.

## Tool inventory

| Tool | Purpose |
|------|---------|
| `set_alarm(hour, minute, label?, days_of_week?)` | Schedule an alarm on the user's phone via the device's Clock app. `hour` 0–23 and `minute` 0–59 are local to the device. Omit `days_of_week` for one-off alarms; supply `[1=Sun ... 7=Sat]` for repeating. Returns `{status: "scheduled", hour, minute, label, days_of_week}`. |

## Wire protocol

When `set_alarm` (or any tool added to `DEVICE_ACTION_TOOLS`)
completes, the orchestrator yields a `DEVICE_ACTION` event in
addition to the normal `tool_call` / `tool_result` pair. Over
`/ws/chat` the WS serializer flattens it to:

```json
{
  "type": "device_action",
  "action": "set_alarm",
  "args": { "hour": 7, "minute": 0, "label": "Standup", "days_of_week": [2,3,4,5,6] },
  "id": "<tool_call_id>",
  "device_id": "<session device name>"
}
```

| Field | Source |
|-------|--------|
| `type` | Always `"device_action"`. |
| `action` | The tool name (`set_alarm` today). |
| `args` | The arguments dict the LLM passed to the tool, verbatim. |
| `id` | The `tool_call.id`, so the companion app can correlate with the `tool_call` / `tool_result` it already received for the same turn. |
| `device_id` | `orchestrator.device_name` — the human-readable label set on the session via the WS first-frame `device_name` or the `X-Device-Name` REST header. May be `null` if the client never set one. |

### Backward compatibility

The companion app's `ChatProtocol` parser drops unknown frame types
through its `ParsedFrame.Unknown` path, so older app builds without
`device_action` support **silently ignore** the event. No agent-side
feature flag or version negotiation is needed.

The event is also a separate frame type (not a new field on
`tool_call` or `tool_result`), so older REST/WS consumers — the
`/api/chat` event log, `/v1/chat/completions`, satellite firmware,
`conversation_db`, `turn_metrics` — see no shape changes.

### What is *not* on the wire

- `device_action` is **not** persisted. Nothing writes it to
  `conversation_histories` or `turn_metrics`. The `tool_call` /
  `tool_result` pair is what survives in conversation history.
- `device_action` is **not** emitted by `/api/chat` or
  `/v1/chat/completions`. Both surfaces drop it: `/api/chat`'s
  REST event filter only forwards a fixed allowlist; `/v1` is
  stateless and returns only the final assistant content.
- The autonomy engine bypasses the session pool, so an autonomy
  turn that called `set_alarm` would yield a `device_action` event
  into the void (no WS subscriber). This is intentional —
  device-side actions only make sense with a live device session.

## Adding a new device action

Adding `set_timer`, `start_navigation`, etc. is a three-line change
plus the platform intent on the companion app:

1. **MCP server.** Declare the new `Tool(...)` in
   `mcp_device_action_tools/mcp_server.py` with the right
   `inputSchema`, add a handler that returns
   `{"status": "<verb>", ...}`, and route the tool name in the
   `call_tool` dispatcher.
2. **Allowlist.** Add the tool name to `DEVICE_ACTION_TOOLS` in
   `selene_agent/orchestrator.py`. This is the gate — without it
   the tool runs but no `device_action` event is emitted.
3. **Companion app.** Teach `DeviceActionDispatcher` and
   `DeviceActionCard` how to render and fire the new action. See
   the wire-protocol doc in the
   [`havencore-companion-app`](https://github.com/ThatMattCat/havencore-companion-app)
   repo for the parsing contract.

The server-side handler does not need to perform the action — that
is the device's job. Keep handlers cheap (validation + status string
only) so the LLM's tool-call latency stays low.

## Configuration

No env vars. The server is spawned via `MCP_SERVERS` in `.env`:

```json
{
  "name": "device_action",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_device_action_tools"],
  "enabled": true
}
```

`MCPClientManager` discovers `set_alarm` automatically on the next
agent start.

## Internals worth knowing

- **The MCP handler is intentionally a no-op.** It returns a status
  blob purely so the LLM has something to read back. The actual
  side-effect is the device firing the intent on receipt of the
  `device_action` frame.
- **Emission is gated by the `DEVICE_ACTION_TOOLS` frozenset.** A
  tool not in the set runs normally (tool_call → tool_result → final
  assistant message) but produces no `device_action` event. This is
  what keeps the surface scoped to genuinely device-targeted tools.
- **One event per tool call.** The orchestrator emits the event
  inside the `for tool_call in assistant_message.tool_calls` loop,
  so a single LLM turn that called `set_alarm` twice would emit
  two `device_action` frames with distinct `id`s.
- **No retry semantics.** The agent fires the event once and
  forgets. If the device is offline or the user is on a different
  satellite, the alarm is not scheduled — the assistant's natural
  reply ("OK, alarm set for 7 AM") will be a lie. This is a known
  limitation; recovering it would require either a return channel
  from the companion app or a server-side fallback (e.g.
  `schedule_reminder` with `channel: "ha_push"`).

## Troubleshooting

### LLM calls `set_alarm` but no alarm appears on the phone

Check the `/ws/chat` event stream for the `device_action` frame:

```bash
docker compose exec -T agent python -c "
from selene_agent.orchestrator import DEVICE_ACTION_TOOLS
print('allowlist:', sorted(DEVICE_ACTION_TOOLS))
"
```

If `set_alarm` isn't listed, the orchestrator is too old for this
feature — pull the latest `improvements` branch.

If the allowlist is correct but the frame still doesn't arrive,
check that the client connected with a `device_name` it owns: the
orchestrator emits `device_id: null` when no label was set, and
older companion-app builds may filter on a non-null device_id.

### Tool isn't surfaced to the LLM

```bash
curl -s http://localhost:6002/api/tools | jq '.servers[] | select(.name=="device_action")'
```

If the server isn't listed, confirm the `device_action` entry is
present in `MCP_SERVERS` (`.env`) and that the agent was restarted
after the env change (env vars are read at container start, not on
`docker compose restart`).

### `set_alarm` fires but with wrong hour

`hour` and `minute` are interpreted as device-local time. The agent
does not translate timezones — if the user says "7 AM Pacific" but
their phone is on Eastern, the alarm fires at 7 AM Eastern. The
companion app has no way to know the user's intended timezone; rely
on the LLM to ask if it's ambiguous.

## Related files

- `services/agent/selene_agent/modules/mcp_device_action_tools/mcp_server.py`
  — implementation.
- `services/agent/selene_agent/orchestrator.py` — `DEVICE_ACTION`
  EventType, `DEVICE_ACTION_TOOLS` allowlist, emission inside the
  tool-execution loop.
- `services/agent/selene_agent/api/chat.py` — `/ws/chat` serializer
  (generic; flattens any `EventType.value` frame).

## See also

- [Companion App integration](../../../integrations/companion-app.md) —
  the other half of the wire contract, including the existing
  UnifiedPush/ntfy push surface that is independent of
  `device_action` (push wakes the phone in the background;
  `device_action` rides a live chat WebSocket).
- [Reminder tool](reminder.md) — for "remind me at 7 AM" requests
  the device-side `set_alarm` is the right tool; for "remind me to
  take out the trash on Sunday at 6 PM" use `schedule_reminder`
  with `channel: "ha_push"` or `signal` instead. Alarms are blocking
  audio; reminders are background notifications.
