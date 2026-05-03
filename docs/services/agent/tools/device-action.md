# MCP Server: Device Actions (`mcp_device_action_tools`)

Reference doc for the device-side action MCP server. Two families of
tools live here:

- **Fire-and-forget intents** (`set_alarm`). The handler is a no-op
  that returns a structured status; the orchestrator emits a
  `device_action` event after the `tool_result` and the companion app
  fires the matching platform intent (e.g.
  [`AlarmClock.ACTION_SET_ALARM`](https://developer.android.com/reference/android/provider/AlarmClock#ACTION_SET_ALARM)).
- **Camera round-trip tools** (`take_photo`, `identify_object_in_photo`,
  `read_text_from_image`). The orchestrator emits the `device_action`
  *before* the tool body runs, the phone captures + uploads a JPEG to
  `/api/companion/upload`, and the orchestrator awaits the upload
  future in-process. Vision-chained variants then forward the
  resulting `image_url` to the vllm-vision pipeline and surface the
  vision response as the tool result. See [Camera tools](#camera-tools)
  for the full flow.

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_device_action_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_device_action_tools` |
| Transport | MCP stdio |
| Server name | `havencore-device-action-tools` |
| Backend | None for `set_alarm` — handler returns a status string. Camera tools route through the in-process companion-upload registry (see [Camera tools](#camera-tools)); the MCP handlers for those are benign-error fallbacks since the upload future + blob store live in the agent process and are unreachable from this stdio subprocess. |
| Tool count | 4 |
| Wire-protocol allowlist | `selene_agent.orchestrator.DEVICE_ACTION_TOOLS` (a `frozenset`) |
| Pre-execute allowlist | `selene_agent.orchestrator.PRE_EXECUTE_DEVICE_ACTION_TOOLS` — subset whose `device_action` frame ships *before* the tool body runs |
| Companion-upload allowlist | `selene_agent.orchestrator.COMPANION_UPLOAD_TOOLS` — subset whose result is filled by `/api/companion/upload` (orchestrator routes around MCP) |
| Vision-chained map | `selene_agent.orchestrator.VISION_CHAINED_TOOLS` — subset that additionally posts the uploaded image to the vision pipeline |

The execution model is two-sided:

1. **Server side (this module + orchestrator).** For `set_alarm`, the
   MCP handler validates the args and returns a
   `{"status": "scheduled", ...}` JSON blob — purely so the LLM has
   something to read back and phrase a natural reply ("Alarm set for
   7 AM."). No DB writes, no external calls. For camera tools, the
   orchestrator short-circuits the MCP path and runs
   `_handle_companion_camera()` instead: register an `asyncio.Future`
   keyed by `tool_call_id`, await its resolution from the upload
   endpoint, and (for vision-chained tools) call `_call_vision()`
   in-process before returning the JSON tool result.
2. **Wire side (orchestrator → `/ws/chat` → companion app).** When
   the tool name is in `DEVICE_ACTION_TOOLS`, the orchestrator emits
   a [`device_action` event](#wire-protocol). For tools in
   `PRE_EXECUTE_DEVICE_ACTION_TOOLS` (the camera tools) the frame
   ships *before* the tool body runs — otherwise the handler would
   await an upload that depends on a frame that never went out. For
   `set_alarm` the frame still ships right after `tool_result`. The
   companion app's `DeviceActionDispatcher` parses the frame and
   either fires the platform intent (`set_alarm`) or launches the
   camera capture flow (`take_photo` and friends).

The `tool_call` / `tool_result` pair stays — the chat screen still
renders a `ToolCallCard` server-side breadcrumb. The new event is
what actually fires the intent on the device.

## Tool inventory

| Tool | Family | Purpose |
|------|--------|---------|
| `set_alarm(hour, minute, label?, days_of_week?)` | fire-and-forget | Schedule an alarm on the user's phone via the device's Clock app. `hour` 0–23 and `minute` 0–59 are local to the device. Omit `days_of_week` for one-off alarms; supply `[1=Sun ... 7=Sat]` for repeating. Returns `{status: "scheduled", hour, minute, label, days_of_week}`. |
| `take_photo(reason?)` | camera round-trip | Ask the companion app to capture a photo and upload it. Returns `{status: "captured", image_url, mime, captured_at, device_id}`. The `image_url` can then be passed to vision tools (e.g. `query_multimodal_api`) for follow-up analysis — but if the user's intent is identification or OCR, prefer the more specific tools below so the LLM doesn't have to chain. Times out after `COMPANION_PHOTO_UPLOAD_TIMEOUT_SEC` seconds (default 25) with `{status: "timeout", error: "..."}`. |
| `identify_object_in_photo(hint?)` | camera + vision-chained | Capture a photo, then post `image_url` to the vision pipeline with the identify prompt. Returns `{status: "captured_and_analyzed", image_url, captured_at, identification}`. Optional `hint` ("plant", "bird") narrows the prompt. |
| `read_text_from_image()` | camera + vision-chained | Capture a photo, then post `image_url` to the vision pipeline with the OCR prompt (low temperature, 1024 tokens). Returns `{status: "captured_and_analyzed", image_url, captured_at, text}`. |

## Wire protocol

When a tool in `DEVICE_ACTION_TOOLS` runs, the orchestrator yields a
`DEVICE_ACTION` event in addition to the normal `tool_call` /
`tool_result` pair. Over `/ws/chat` the WS serializer flattens it to:

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
| `action` | The tool name (`set_alarm`, `take_photo`, `identify_object_in_photo`, `read_text_from_image`). |
| `args` | The arguments dict the LLM passed to the tool, verbatim. |
| `id` | The `tool_call.id`, so the companion app can correlate with the `tool_call` / `tool_result` it already received for the same turn. For camera tools this is also the key the upload future is registered under (the phone POSTs it back as `tool_call_id` in the multipart body). |
| `device_id` | `orchestrator.device_name` — the human-readable label set on the session via the WS first-frame `device_name` or the `X-Device-Name` REST header. May be `null` if the client never set one. |

### Emission timing

| Tool family | When the frame ships | Why |
|---|---|---|
| `set_alarm` (and any future fire-and-forget tool added to `DEVICE_ACTION_TOOLS` only) | After `tool_result` | The LLM-visible result is the trigger; no round trip is needed. |
| Camera tools (in `PRE_EXECUTE_DEVICE_ACTION_TOOLS`) | Before the tool body runs | The handler awaits an upload future that depends on the phone seeing the frame; emitting after would deadlock until timeout. |

The orchestrator's tool loop emits at most one `device_action` per
`tool_call`; the two timing branches are mutually exclusive.

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

## Camera tools

Camera tools are a distinct family from `set_alarm`: instead of
firing a platform intent and forgetting, they round-trip a captured
JPEG back to the agent and (for vision-chained variants) chain to
the in-process vision pipeline before returning a result to the
LLM. The agent never holds the bytes longer than `COMPANION_BLOB_TTL_SEC`.

### End-to-end flow

```
LLM → take_photo / identify_object_in_photo / read_text_from_image
    ↓
orchestrator
  • emits DEVICE_ACTION frame on /ws/chat (PRE-execute)
  • registers asyncio.Future in api/companion._pending_uploads[tool_call_id]
  • awaits the future (timeout: COMPANION_PHOTO_UPLOAD_TIMEOUT_SEC)
    ↓
companion app
  • DeviceActionDispatcher launches CaptureActivity
  • user takes photo → JPEG written to FileProvider URI in app cache
  • CompanionUploadApi POSTs multipart to /api/companion/upload
    ↓
api/companion.py
  • stashes bytes in BlobStore (TTL + byte-cap eviction)
  • mints token → image_url = http://agent:6002/api/companion/blob/<token>
  • resolves the pending future with {image_url, mime, captured_at, ...}
    ↓
orchestrator (await returns)
  • take_photo: returns {status: "captured", image_url, ...} to LLM
  • identify_object_in_photo / read_text_from_image:
      calls api/vision._call_vision(image_url, prompt) directly in-process
      returns {status: "captured_and_analyzed", image_url, identification|text}
```

### Why the upload future lives in the agent process, not this MCP module

MCP servers run as stdio subprocesses spawned by `MCPClientManager`
(see [development.md](development.md)). A `dict[str, asyncio.Future]`
inside this module would be unreachable from the agent's
`/api/companion/upload` HTTP handler. The orchestrator therefore
short-circuits MCP for tools in `COMPANION_UPLOAD_TOOLS`: it routes
them through `_handle_companion_camera()`, which keeps the future
registry and blob store in the agent process where the upload
endpoint lives. The MCP `take_photo` / `identify_object_in_photo` /
`read_text_from_image` declarations still exist so the LLM
discovers the tools via the normal MCP `tools/list` handshake; their
handlers are benign-error fallbacks in case anything ever invokes
the MCP path directly.

### Vision chaining

`VISION_CHAINED_TOOLS` in `orchestrator.py` maps each chained tool to
the prompt it should send to the vision pipeline:

| Tool | Prompt source | `max_tokens` | `temperature` | Result key |
|---|---|---|---|---|
| `identify_object_in_photo` | `_DEFAULT_IDENTIFY_PROMPT` (+ optional `hint`) | 300 | 0.7 | `identification` |
| `read_text_from_image` | `_DEFAULT_OCR_PROMPT` | 1024 | 0.1 | `text` |

Prompts are duplicated from
`mcp_vision_tools.server.DEFAULT_IDENTIFY_PROMPT` /
`DEFAULT_OCR_PROMPT` rather than imported because importing that
module would side-effect a stdio MCP server. Keep the two copies in
sync if you tune either prompt.

The chain calls `selene_agent.api.vision._call_vision()` directly —
no self-HTTP — so the latency cost of chaining is one extra vllm
call (no extra hop). Errors from the vision call bubble up as
`{status: "vision_error", error, image_url}` so the LLM can still
phrase a useful reply about what happened.

### Companion-app capability gating

The companion app's Settings → Camera tools card has a master toggle
(gates all camera tools) plus per-tool toggles for the
vision-chained variants. When a toggle is off, the dispatcher
returns `DeviceActionResult.Disabled` *before* launching the camera —
no capture, no upload, no agent-visible side effect. The agent's
upload future then times out at `COMPANION_PHOTO_UPLOAD_TIMEOUT_SEC`
and the LLM sees `{status: "timeout", ...}`. (Server-side
short-circuiting based on a phone-shipped `companion_capabilities`
bitmap is a Step-6 enhancement; in v1 the gating lives entirely on
the phone.)

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

### Adding a new camera-style tool

Use this variant when the new tool needs a fresh photo from the
user's phone (and optionally a vision-pipeline pass on the result):

1. **MCP server.** Declare the `Tool(...)` and route it in
   `call_tool` to `companion_camera_fallback(name)`.
2. **Allowlists.** Add the tool name to all three of
   `DEVICE_ACTION_TOOLS`, `PRE_EXECUTE_DEVICE_ACTION_TOOLS`, and
   `COMPANION_UPLOAD_TOOLS`.
3. **Vision chain (optional).** If the tool should run vision on the
   captured image before returning to the LLM, add an entry to
   `VISION_CHAINED_TOOLS` with the prompt builder, `max_tokens`,
   `temperature`, and result key. Otherwise it returns the raw
   `{status: "captured", image_url, ...}` (like `take_photo`).
4. **Companion app.** Add a `DeviceAction.CameraCapture` sealed
   variant + parser branch + a `DeviceActionCard` arm + a per-tool
   toggle in `SettingsRepository` and `CompanionCameraCard`. The
   dispatcher's `captureAndUpload()` already handles the capture +
   upload primitive — only the per-tool toggle gate and the chat
   card presentation differ per variant.
5. **Tests.** `tests/test_companion_vision_chain.py::test_camera_tool_constants_aligned`
   checks the four allowlists are mutually consistent; extend the
   chain assertions in the same file for any new vision-chained tool.

## Configuration

The MCP server is spawned via `MCP_SERVERS` in `.env`:

```json
{
  "name": "device_action",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_device_action_tools"],
  "enabled": true
}
```

`MCPClientManager` discovers all four tools automatically on the next
agent start.

The camera tools additionally read these env vars (see
[configuration.md](../../../configuration.md#companion-app-camera-tools)
for the full reference):

| Var | Default | Purpose |
|---|---|---|
| `COMPANION_PHOTO_UPLOAD_TIMEOUT_SEC` | `25` | How long the orchestrator waits on the upload future before returning `{status: "timeout", ...}` to the LLM. 25s comfortably covers camera launch + capture + multipart upload on LAN. |
| `COMPANION_BLOB_TTL_SEC` | `600` | How long an uploaded blob stays fetchable at `/api/companion/blob/<token>`. Vision/face pipelines fetch within seconds; the rest of the TTL absorbs retries and dashboard previews. |
| `COMPANION_BLOB_MAX_BYTES` | `10485760` | Total in-memory cap for `BlobStore`. Oldest entries evicted first when breached. 10 MiB ≈ ~5 phone-grade JPEGs. Per-upload size limit is the same. |

## Internals worth knowing

- **The `set_alarm` MCP handler is intentionally a no-op.** It
  returns a status blob purely so the LLM has something to read
  back. The actual side-effect is the device firing the intent on
  receipt of the `device_action` frame.
- **Camera-tool MCP handlers are benign-error fallbacks.** The
  orchestrator routes `take_photo`, `identify_object_in_photo`, and
  `read_text_from_image` around MCP entirely (see [Camera tools →
  Why the upload future lives in the agent process](#why-the-upload-future-lives-in-the-agent-process-not-this-mcp-module)).
  The MCP handlers exist so the tool declarations surface to the LLM
  via the normal MCP `tools/list` discovery path; they return a
  structured error if anything ever invokes them directly.
- **Emission is gated by the `DEVICE_ACTION_TOOLS` frozenset.** A
  tool not in the set runs normally (tool_call → tool_result → final
  assistant message) but produces no `device_action` event. This is
  what keeps the surface scoped to genuinely device-targeted tools.
- **Pre-execute emission is gated by `PRE_EXECUTE_DEVICE_ACTION_TOOLS`.**
  Tools in this subset emit before the body runs — required for the
  camera round-trip to work. `set_alarm` stays on the post-result
  path.
- **One event per tool call.** The orchestrator emits the event
  inside the `for tool_call in assistant_message.tool_calls` loop,
  so a single LLM turn that called `set_alarm` twice would emit
  two `device_action` frames with distinct `id`s. The same is true
  for camera tools — each `tool_call.id` registers its own upload
  future under its own key.
- **`current_tool_call_id` ContextVar.** The orchestrator sets a
  module-level `contextvars.ContextVar` around `_execute_tool_call`
  for the duration of each tool invocation. In-process helpers can
  read it to correlate without threading the id through every layer.
  Out-of-process MCP subprocess handlers cannot read it — they would
  receive the id via tool arguments instead. Today only the camera
  path uses this hook.
- **No retry semantics.** The agent fires `set_alarm`'s event once
  and forgets. If the device is offline or the user is on a
  different satellite, the alarm is not scheduled — the assistant's
  natural reply ("OK, alarm set for 7 AM") will be a lie. Camera
  tools fail more loudly: the upload never arrives, the future
  times out, and the LLM sees `{status: "timeout", ...}` and
  apologizes. Recovering `set_alarm` would require either a return
  channel from the companion app or a server-side fallback (e.g.
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
  — Tool declarations + `set_alarm` handler + `companion_camera_fallback`.
- `services/agent/selene_agent/orchestrator.py` — `DEVICE_ACTION`
  EventType, the four allowlists (`DEVICE_ACTION_TOOLS`,
  `PRE_EXECUTE_DEVICE_ACTION_TOOLS`, `COMPANION_UPLOAD_TOOLS`,
  `VISION_CHAINED_TOOLS`), `_handle_companion_camera()`,
  `_await_companion_upload_payload()`, `_ask_vision()`, and the
  pre/post emission branches inside the tool-execution loop.
- `services/agent/selene_agent/api/companion.py` —
  `POST /api/companion/upload`, `GET /api/companion/blob/{token}`,
  `BlobStore` (TTL + byte-cap eviction), and the
  `_pending_uploads` future registry.
- `services/agent/selene_agent/api/vision.py` — `_call_vision()`,
  the chokepoint the vision-chain branch calls into directly.
- `services/agent/selene_agent/api/chat.py` — `/ws/chat` serializer
  (generic; flattens any `EventType.value` frame).
- `services/agent/tests/test_companion_upload.py` — TTL, byte-cap,
  410-on-no-future, and happy-path multipart coverage.
- `services/agent/tests/test_companion_vision_chain.py` — vision
  chain assertions (right prompt + params per tool, take_photo
  doesn't chain, vision errors → structured envelope, allowlist
  alignment guard).

## See also

- [Companion App integration](../../../integrations/companion-app.md) —
  the other half of the wire contract, including the existing
  UnifiedPush/ntfy push surface that is independent of
  `device_action` (push wakes the phone in the background;
  `device_action` rides a live chat WebSocket).
- [API reference → Companion-app uploads](../../../api-reference.md#companion-app-uploads) —
  endpoint shapes for `/api/companion/upload` and
  `/api/companion/blob/{token}`.
- [Vision tools](vision.md) — the existing
  `describe_image` / `identify_object` / `read_text_in_image` MCP
  tools operate on a URL the LLM already has; the camera tools here
  produce that URL by capturing a fresh photo from the phone.
- [Reminder tool](reminder.md) — for "remind me at 7 AM" requests
  the device-side `set_alarm` is the right tool; for "remind me to
  take out the trash on Sunday at 6 PM" use `schedule_reminder`
  with `channel: "ha_push"` or `signal` instead. Alarms are blocking
  audio; reminders are background notifications.
