# Companion App (Android push notifications)

How HavenCore wakes the user's phone with autonomy briefings, anomaly
alerts, reminders, and ad-hoc agent messages — without keeping a
WebSocket open. The phone runs the
[`havencore-companion-app`](https://github.com/ThatMattCat/havencore-companion-app),
which uses [UnifiedPush](https://unifiedpush.org/) with the user's
chosen distributor (typically the [ntfy](https://ntfy.sh/) Android
app). This page documents the agent-side protocol; the companion app's
own README covers its UI and Settings surface.

## Why UnifiedPush + ntfy

- **Self-hosted ethos.** The user can run their own ntfy server on the
  same LAN as HavenCore. No Google or Apple servers in the path.
- **Distributor-agnostic on the client.** UnifiedPush is a spec, not a
  vendor — the companion app accepts whatever distributor the user
  installs (ntfy, NextPush, FCMUP, …). The agent doesn't care which
  one produced the endpoint, only that an endpoint URL exists.
- **Survives Doze and app-killed states.** The distributor holds the
  long-lived socket as a foreground service; when a payload arrives,
  Android wakes the companion app via broadcast. The companion app
  itself has no foreground service.
- **No agent-side ntfy URL config.** The endpoint URL the distributor
  returns is fully self-contained — the agent stores it verbatim and
  POSTs to it directly. Deployments are free to point different
  devices at different ntfy servers; the agent just iterates the
  registered list.

## End-to-end flow

```
phone Settings → "Enable notifications"
  → UnifiedPush.registerApp(ctx)   (distributor returns endpoint URL)
  → companion app POSTs /api/push/register { device_id, device_label, endpoint }
  → agent stores row in push_devices

… later, autonomy engine fires …

  → handler result has _notify_channel = "ntfy"
  → engine._build_notifier("ntfy", …) returns NtfyFanoutNotifier
  → NtfyFanoutNotifier reads push_devices  (every device gets one POST)
  → for each row: NtfyNotifier POSTs the wire envelope to row.endpoint
  → ntfy server forwards bytes to ntfy Android app
  → ntfy distributor broadcasts org.unifiedpush.android.connector.MESSAGE
  → companion app's PushReceiver decodes JSON, posts NotificationCompat
  → tap deep-links to chat?sessionId=<id> via PendingIntent
```

## Wire format

Each autonomy fire produces one JSON envelope, POSTed once per
registered endpoint. UnifiedPush hard-caps payload at 4096 bytes;
`NtfyNotifier` truncates `body` to 3000 chars defensively before
serializing.

```json
{
  "v": 1,
  "type": "autonomy_brief" | "anomaly" | "reminder" | "act_confirm" | "ad_hoc",
  "title": "Selene",
  "body": "<= 3000 chars",
  "session_id": "abc-123",
  "severity": "none" | "info" | "warn" | "alert"
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `v` | yes | Schema version. The companion app's decoder uses `ignoreUnknownKeys = true`; bump only on breaking changes. |
| `type` | yes | Surface category. Reserved for a future per-type Android notification channel split (`autonomy_brief` vs `anomaly` vs `reminder` vs `act_confirm`); v1 routes everything through one `havencore_autonomy` channel. |
| `title` | yes | Notification title. Defaults to `AGENT_NAME` (typically "Selene") when the agent omits it. |
| `body` | yes | Notification body. Truncated to 3000 chars; the companion app renders with `BigTextStyle` for the expanded view. |
| `session_id` | no | If set, tapping the notification deep-links to that chat session in the companion app (cold-resume via `/api/conversations/{session_id}/resume`). |
| `severity` | yes | Maps phone-side to `NotificationCompat.PRIORITY_*`: `none`/`info` → DEFAULT, `warn` → HIGH + short vibration, `alert` → HIGH + long pulsing vibration. v1 respects system Do-Not-Disturb (no DND bypass). |

Bearer auth: if `NTFY_PUBLISH_TOKEN` is set in the agent's `.env`, the
agent presents `Authorization: Bearer <token>` on every POST. Default
empty for self-hosted ntfy with no auth.

## Agent endpoints

LAN-only, unauthenticated (consistent with the rest of `/api/*`).
Backed by the `push_devices` Postgres table (`device_id` UUID primary
key, `device_label`, `endpoint`, `platform`, `registered_at`,
`last_seen_at`).

```
POST /api/push/register
  body: { "device_id": "<uuid>", "device_label": "Matt's S24",
          "endpoint": "https://ntfy.example.com/UPxxxxxxxxxxxx",
          "platform": "android" }
  -> 200 { "ok": true }    (always upsert; rotated endpoints replace prior row)
  -> 400 if endpoint not http/https or device_id not a UUID

DELETE /api/push/register/{device_id}
  -> 200 { "ok": true }
  -> 404 if not present     (companion app treats as success)

GET /api/push/register
  -> 200 { "devices": [
      { device_id, device_label, endpoint, platform,
        registered_at, last_seen_at }, ... ] }
```

Re-registering the same `device_id` upserts in place (updates
`device_label`, `endpoint`, and `last_seen_at`); endpoint rotation by
the distributor never produces duplicate rows.

## Opting an autonomy item into push delivery

Push is a *channel*, surfaced through the same `_notify_channel`
discriminator the engine uses for Signal, HA push, and speaker.
Anomaly/watch/`watch_llm` handlers can return a result dict with:

```python
return {
    # ... existing keys (status, summary, severity, _unusual, …)
    "_notify_channel": "ntfy",
    "_notify_title":   "Selene",
    "_notify_body":    "<your message>",
    "_notify_cfg": {
        "ntfy_session_id": "<existing-session-id>",   # optional, for tap deep-link
        "ntfy_type":       "anomaly",                  # optional, default "ad_hoc"
    },
}
```

For confirmation notifications, set `deliver.channel = "ntfy"` on the
item's config.

`NtfyFanoutNotifier` reads the `push_devices` table at send-time, so
devices added between dispatches are picked up on the next fire. If
no devices are registered, the notifier logs `[NtfyFanout] no
registered devices; dropping notification` and the
`autonomy_runs.notified_via` for that run is `null`.

`send()` returns `True` if at least one endpoint accepted; per-endpoint
4xx/5xx responses are logged but do not fail the autonomy run. The
engine stamps `notified_via = "ntfy"` whenever any endpoint succeeded.

## Setup walkthrough

The phone-side steps live in the
[`havencore-companion-app`](https://github.com/ThatMattCat/havencore-companion-app)
README. Agent-side setup is minimal:

1. **Run an ntfy server.** The `compose.yaml` ships a `ntfy` service
   on port `8585` (`binwiederhier/ntfy`, no auth, LAN-only). Brought
   up automatically with the rest of the stack:
   ```bash
   docker compose up -d ntfy
   ```
   The web UI is at `http://<HOST_IP_ADDRESS>:8585/`. Nginx also
   provides a memorability redirect — `http://<HOST_IP_ADDRESS>/ntfy`
   (and `/ntfy/`) bounces to `:8585`. Note that ntfy refuses
   sub-path hosting, so all real traffic (phone-side ntfy app,
   UnifiedPush endpoints, agent publishes) uses the `:8585` URL
   directly; the `/ntfy` redirect is humans-typing-in-browsers only.
   - Public ntfy.sh works as an alternative without any setup — fine
     for testing, not recommended for sensitive notifications since
     topic names are guessable URLs.
2. **(Optional) Set `NTFY_PUBLISH_TOKEN` in `.env`** if you've
   configured your ntfy server to require bearer auth on publish.
   Skip if your ntfy is no-auth (the default for the shipped service).
3. **Restart the agent stack** if you changed `NTFY_PUBLISH_TOKEN`
   (env vars are read at container start; volume-mounted code reloads
   on `docker compose restart agent`).
4. **Install ntfy on the phone** (F-Droid, Play Store, or
   `https://ntfy.sh/app`), open ntfy → ⋮ → **Settings** → **Default
   server** → enter `http://<HOST_IP_ADDRESS>:8585` (plain http; the
   app warns, accept). Then exempt the ntfy app from battery
   optimization (Samsung/OEM devices aggressively kill background
   apps that aren't exempt — without this the WebSocket gets killed
   and pushes stop arriving when the phone idles).
5. **In the companion app's Settings**, toggle "Enable notifications".
   The app calls `UnifiedPush.registerApp(ctx)`, the distributor
   returns an endpoint URL like `http://<HOST_IP_ADDRESS>:8585/UPxxxx`,
   and the app POSTs it to the agent's `/api/push/register`. Confirm
   in the agent log:
   ```
   [push] registered device=<uuid> label='<phone>' endpoint=http://<HOST_IP_ADDRESS>:8585/UPxxxx
   ```
6. **Verify with a manual fanout** (no autonomy fire required):
   ```bash
   docker compose exec -T agent python -c "
   import asyncio
   from selene_agent.utils.conversation_db import conversation_db
   from selene_agent.autonomy.notifiers import NtfyFanoutNotifier
   async def main():
       await conversation_db.initialize()
       n = NtfyFanoutNotifier()
       print(await n.send(title='Selene', body='hello phone', severity='info'))
       await conversation_db.close()
   asyncio.run(main())
   "
   ```
   Phone notification should appear within ~1 s.

## Verification

| Question | Where to look |
|----------|---------------|
| Is the phone registered? | `curl http://localhost:6002/api/push/register` — your `device_id` should be in `devices[]`. |
| Did the agent receive the registration POST? | `docker compose logs agent | grep "\[push\]"` |
| Did an autonomy run fan out? | `autonomy_runs.notified_via = "ntfy"` for the run, plus `[NtfyFanout] delivered to N/M device(s)` in agent logs. |
| Did the phone receive the broadcast? | logcat: `adb logcat | grep -E 'Push:Recv|Push:Reg'` (companion app's tags). |

## Limitations / out of scope

- **Plaintext over TLS.** The wire envelope is plain JSON; security
  rests on the distributor's HTTPS connection to its ntfy server.
  VAPID-style end-to-end encryption between the agent and the
  companion app is reserved for a v2 once remote (non-LAN) access is
  real and the threat model demands it.
- **No notification action buttons in v1.** Reply / Mark-done /
  Dismiss-all require server-side action handling tied to the
  autonomy `act` tier — its own design problem, deferred.
- **iOS / APNS not supported.** Android-only; iOS would need a
  separate notifier (APNS doesn't speak UnifiedPush).
- **No per-device auth.** All endpoints share the single
  `NTFY_PUBLISH_TOKEN` (or no token). Per-endpoint auth (a future
  `push_devices.auth` JSONB column) is a v2 concern.
- **No stale-row pruning.** The `push_devices` table grows with each
  `device_id` ever registered. Manual `DELETE /api/push/register/<id>`
  works; an automatic `last_seen_at < now() - 30d` sweep is
  unimplemented.
- **Briefing handler stays on Signal.** The autonomy briefing handler
  hard-codes `SignalNotifier`; it does not honor `_notify_channel`
  yet. Routing briefings via ntfy would require a one-line change to
  `briefing.py` to use `_build_notifier` — out of scope for now.

## Device-side actions (live `/ws/chat` channel)

Independent from the UnifiedPush/ntfy push surface above, the
companion app also listens on `/ws/chat` for `device_action` frames.
This is the path used when the user is *actively chatting* and the
LLM decides to schedule something on the device itself — the
canonical example is "set an alarm for 7 AM," which fires
[`AlarmClock.ACTION_SET_ALARM`](https://developer.android.com/reference/android/provider/AlarmClock#ACTION_SET_ALARM)
through the device's Clock app.

Surface comparison:

| | UnifiedPush + ntfy | `device_action` over `/ws/chat` |
|---|---|---|
| Triggered by | autonomy engine fires | LLM tool call inside a live chat turn |
| Transport | HTTPS POST → ntfy → distributor → Android broadcast | already-open WebSocket |
| Phone state | works while app is killed/Doze'd | requires the app to be foreground or holding the WS open (assist-slot voice overlay or Chat screen) |
| Wakes the screen | yes (notification) | no — fires the intent silently (camera tools do open the camera app) |
| Payload type | `autonomy_brief` / `anomaly` / `reminder` / `act_confirm` / `ad_hoc` | `set_alarm`, `take_photo`, `identify_object_in_photo`, `read_text_from_image`, `who_is_in_view` (extensible — see below) |

Wire-protocol shape (sent verbatim by the agent's WS serializer):

```json
{
  "type": "device_action",
  "action": "set_alarm",
  "args": { "hour": 7, "minute": 0, "label": "Standup", "days_of_week": [2,3,4,5,6] },
  "id": "<tool_call_id>",
  "device_id": "<session device name>"
}
```

The `tool_call` / `tool_result` pair stays on the wire as the
server-side breadcrumb — the chat screen still renders a
`ToolCallCard`. The `device_action` frame is what actually fires
the platform intent, dispatched by the companion app's
`DeviceActionDispatcher`.

**Emission timing differs by tool family.** `set_alarm` ships its
`device_action` frame *after* the `tool_result` — a fire-and-forget
intent. The camera tools below ship the frame *before* the tool body
runs (orchestrator's `PRE_EXECUTE_DEVICE_ACTION_TOOLS` allowlist),
because their handler awaits an upload from the phone that depends on
the frame having already gone out.

**Backward compatibility.** Older companion-app builds without
`device_action` support drop the frame silently via
`ChatProtocol.ParsedFrame.Unknown` — no version negotiation needed.
Unknown action names render an inert "(unsupported)" card on
current builds.

### Camera round-trip tools

`take_photo`, `identify_object_in_photo`, `read_text_from_image`,
and `who_is_in_view` extend the `device_action` channel with a
phone-to-agent return path: the phone POSTs the captured JPEG back
to `/api/companion/upload`, the agent stashes it in an in-memory
`BlobStore`, mints a short-lived `image_url`, and resolves a
`tool_call_id`-keyed `asyncio.Future` so the orchestrator's awaiting
`_handle_companion_camera()` can return a structured result to the
LLM (or chain to the vision pipeline / face recognition first).

End-to-end flow:

```
LLM tool_call → orchestrator emits DEVICE_ACTION (pre-execute)
              → companion app: DeviceActionDispatcher launches CaptureActivity
              → user takes photo → JPEG written to FileProvider URI in cache
              → CompanionUploadApi POSTs multipart to /api/companion/upload
              → BlobStore stashes bytes, mints image_url, resolves future
              → orchestrator returns to LLM:
                 take_photo                  → {status: "captured", image_url, ...}
                 identify_object_in_photo    → {status: "captured_and_analyzed", identification}
                 read_text_from_image        → {status: "captured_and_analyzed", text}
                 who_is_in_view              → {status: "captured_and_recognized", found, name?, ...}
```

The agent-side details (allowlists, vision-chain map, prompt
duplication, why the upload future lives in the agent process and not
in the MCP module) are documented in
[device-action.md → Camera tools](../services/agent/tools/device-action.md#camera-tools).
The upload + blob endpoints are documented in
[api-reference.md → Companion-app uploads](../api-reference.md#companion-app-uploads).
Tunable env vars (timeout, TTL, byte cap) live in
[configuration.md → Companion-app camera tools](../configuration.md#companion-app-camera-tools).

**Capability gating.** The companion app's Settings → Camera tools
card has a master toggle plus per-tool toggles for each chained
variant (identify, OCR, face recognition). When a toggle is off,
the dispatcher returns `DeviceActionResult.Disabled` *before*
launching the camera —
no capture, no upload, no agent-visible side effect — and the
agent's upload future eventually times out. Server-side
short-circuiting based on a phone-shipped capabilities bitmap is
deferred to a later step; in v1 the gating lives entirely on the
phone.

**Bug fixed during initial bring-up.** `CaptureActivity` originally
declared `android:noHistory="true"`; that's a footgun for activities
that delegate to a foreign app via `ActivityResultContracts`, because
launching the camera Intent counts as "navigating away" and Android
destroys the activity before the result returns. Symptom is a silent
upload timeout. `excludeFromRecents` + a translucent theme give the
"invisible / not in recents" effect without the `noHistory` side
effect; we call `finish()` manually after the result lands.

**Extending the action set.** A new device-side action (`set_timer`,
`start_navigation`, …) is one MCP tool plus one entry in
`orchestrator.DEVICE_ACTION_TOOLS` plus the matching
`DeviceActionDispatcher` branch on the phone. For a new
camera-style tool that needs a photo (and optionally a vision-pipeline
pass), there are two extra allowlists to update — full walkthrough in
the
[device-actions tool reference](../services/agent/tools/device-action.md#adding-a-new-device-action).

## See also

- [Device actions tool](../services/agent/tools/device-action.md) —
  the agent-side MCP tool catalog and the wire-protocol contract for
  the live `/ws/chat` channel above.
- [Autonomy engine](../services/agent/autonomy/README.md) — the
  notifier abstraction, the `_notify_channel` discriminator, and the
  channels table.
- [API reference → Push registration](../api-reference.md#push-registration-companion-app) —
  endpoint shapes.
- [Configuration → notification targets](../configuration.md) —
  `NTFY_PUBLISH_TOKEN` and friends.
- [`havencore-companion-app`](https://github.com/ThatMattCat/havencore-companion-app)
  — the Android client.
