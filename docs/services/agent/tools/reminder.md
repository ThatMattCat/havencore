# MCP Server: Reminder (`mcp_reminder_tools`)

Reference doc for the reminder-scheduling MCP server. Lets the agent
create, list, and cancel one-shot or recurring reminders that the
autonomy engine fires on schedule and delivers via Signal / Home
Assistant push / speaker TTS.

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_reminder_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_reminder_tools` |
| Transport | MCP stdio |
| Server name | `havencore-reminder-tools` |
| Backend | Local autonomy REST API at `http://localhost:6002/api/autonomy/items` |
| Storage | Reuses the existing `agenda_items` table (`kind='reminder'`) |
| Delivery handler | `selene_agent/autonomy/handlers/reminder.py` |
| Tool count | 3 |

The server is a thin wrapper. The autonomy engine already supports the
`reminder` agenda kind end-to-end: cron validation, persistence,
scheduling, quiet-hours gating, and dispatch through the existing
notifier abstraction. This module's only job is to expose
`schedule_reminder` / `list_reminders` / `cancel_reminder` to the LLM
so any "remind me to X" request produces a real scheduled item.

Going through the local REST API (rather than calling `autonomy_db`
directly from the subprocess) reuses the existing pydantic validation
and triggers `engine.notify_agenda_changed()` so new items are picked
up immediately rather than on the next dispatch tick.

## Tool inventory

| Tool | Purpose |
|------|---------|
| `schedule_reminder(title, body?, in_seconds? \| at? \| cron?, channel?, to?, personalize?)` | Create a reminder. Provide exactly one of `in_seconds` (relative delay), `at` (ISO 8601 absolute time, naive interpreted in `CURRENT_TIMEZONE`), or `cron` (5-field cron expression). `in_seconds` and `at` always produce one-shot items; `cron` produces a recurring item. Default `channel` is `signal`. `personalize` defaults to `true` — see [Personalization](#personalization). Returns `{id, title, channel, cron, one_shot, personalize, next_fire_at}`. |
| `list_reminders(include_disabled?)` | Return active reminders with `id`, `title`, `cron`, `next_fire_at`, `channel`, `one_shot`, and `personalize`. Default skips disabled rows; `include_disabled=true` returns all `kind='reminder'` items. |
| `cancel_reminder(id)` | Hard-delete a reminder by id. Use `list_reminders` first to look up the id. Returns `{deleted: true, id}` or surfaces the API's 404. |

## Time-spec resolution

`schedule_reminder` accepts three mutually exclusive time specs and
normalizes them all into a 5-field cron expression on the `agenda_items`
row:

- **`in_seconds: 90`** → take `now + 90s`, round up to the next minute
  boundary, format as a one-shot cron `M H D MO *`. `one_shot=true`.
- **`at: "2026-04-27T18:00:00"`** → parse ISO 8601 (naive interpreted
  in `CURRENT_TIMEZONE`, `Z` suffix accepted), format as a one-shot
  cron. `one_shot=true`. Past timestamps are rejected.
- **`cron: "0 18 * * 0"`** → pass through unchanged. `one_shot=false`.

The cron *shape* is not what marks an item as one-shot — the explicit
`one_shot` flag in `config` is. Date-style crons used for one-shots
would otherwise re-fire next year; the flag is what tells the engine to
delete the row after the first successful fire.

## Channels

| Channel | Backed by | When to use |
|---------|-----------|-------------|
| `signal` *(default)* | `SignalNotifier` → `send_signal_message` MCP tool | Cross-location reach. Reliable when the user is away from home. |
| `ha_push` | `HAPushNotifier` → `ha_send_notification` MCP tool | Phone/mobile push via Home Assistant's notify service. |
| `speaker` | `SpeakerNotifier` → Kokoro TTS + `mass_play_announcement` | Voice announcement on a Music Assistant target. Use for in-home reminders. |

The `to` argument overrides the per-channel recipient (Signal phone
number, HA `notify.<service>` target, or Music Assistant device name).
When omitted, the defaults from the autonomy notifier layer apply —
see [autonomy/README.md → Configuration](../autonomy/README.md#configuration).

## Personalization

By default (`personalize: true`), every reminder is rewritten by the LLM at
fire time so it lands in Selene's voice rather than the user's verbatim
phrasing. The rewrite is a single direct
`client.chat.completions.create()` call — modeled on
`autonomy/memory_clustering.summarize_cluster` — not a full
`AutonomousTurn`. Implementation lives in
`services/agent/selene_agent/autonomy/reminder_personalize.py`.

The helper returns `{body, image_prompt}`:

- `body` — the rewritten reminder text. Always non-empty; on any failure
  (LLM timeout, JSON parse error, empty response) the helper returns the
  original body so the handler can fall through to deterministic delivery.
- `image_prompt` — only populated when `channel == "signal"` and the
  model judged a small illustrative image would add value. For other
  channels the helper strips this field defensively, even if the model
  ignored the system prompt.

### Image attachments (Signal only)

When the personalization step yields an `image_prompt` and the channel is
`signal`, the reminder handler invokes the `generate_image` MCP tool from
`mcp_general_tools` with that prompt, takes the local file path from the
returned `images[0].path`, and passes it through to
`SignalNotifier.send(..., attachments=[path])`. ComfyUI image generation
is bounded by a 25-second timeout (`IMAGE_GEN_TIMEOUT_SEC` in
`handlers/reminder.py`); if it times out or errors, the handler falls
through to a text-only Signal message with the personalized body.

`SIGNAL_MAX_ATTACHMENT_BYTES = 95 MiB` is enforced by the Signal tool
itself, so reminder images don't need separate size validation.

### Opting out

Pass `personalize: false` to `schedule_reminder` for verbatim delivery —
useful when the user explicitly says *"remind me of my exact words"* or
when the reminder is already in the desired voice. The dashboard
AgendaForm exposes the same toggle.

Existing reminders created before personalization landed have no
`personalize` field in their stored config; both the handler and
`list_reminders` treat that as `personalize=true`, so they get
personalized on their next fire automatically. To preserve verbatim
delivery for one of these legacy items, edit it via the dashboard and
uncheck Personalize.

### Cost / latency

- Personalization adds one LLM round-trip (~100–300 ms on local vLLM)
  per fire.
- Image generation, when invoked, adds 15–30 s end-to-end (ComfyUI
  diffusion). Reminders are inherently asynchronous push notifications,
  so this is acceptable for the in-home use case. Manual triggers via
  `POST /api/autonomy/trigger/{id}` exhibit the same latency.

## One-shot lifecycle

When a one-shot reminder fires successfully, the engine **hard-deletes**
the agenda row so it disappears from the dashboard's Agenda Items list.
The audit trail is preserved: the corresponding `autonomy_runs` row
stays, with its `agenda_item_id` set to NULL via the schema's
`ON DELETE SET NULL` foreign key. The Runs feed therefore still shows
the reminder fired even though the parent agenda item is gone.

The delete is signaled by the reminder handler returning
`_delete_after_run: true` in its result, and is performed by the engine
*after* `insert_run` so the FK reference is valid at insert time.
Recurring reminders (those scheduled with `cron`) are unaffected — they
keep firing until cancelled via `cancel_reminder`.

## Configuration

| Var | Default | Purpose |
|-----|---------|---------|
| `AGENT_API_BASE` | `http://localhost:6002` | Base URL the subprocess uses to reach the autonomy REST API. The default works because the MCP subprocess runs inside the same container as the FastAPI server. |
| `CURRENT_TIMEZONE` | `UTC` | Used to interpret naive ISO timestamps and to resolve cron expressions for `_one_shot_cron` formatting. Read from the same env var as the rest of the agent. |

The agent spawns the server via `MCP_SERVERS` in `.env`:

```json
{
  "name": "reminder",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_reminder_tools"],
  "enabled": true
}
```

Default delivery channel and recipient resolution are not configured
here — they live in the autonomy notifier layer
(`AUTONOMY_BRIEFING_NOTIFY_TO`, `AUTONOMY_HA_NOTIFY_TARGET`,
`SIGNAL_DEFAULT_RECIPIENT`).

## Dashboard

Reminders show up in the agenda items table on `/autonomy` like any
other agenda kind. Click **New** (or **Edit** on an existing reminder)
to open the modal; with `kind=reminder` selected the schedule input is
a friendly time picker rather than a raw cron field:

- **Easy** mode (default) has three sub-tabs:
  - **In…** — number + unit (minutes/hours/days) → one-shot relative
  - **At…** — date and time pickers → one-shot at an absolute moment
  - **Every…** — frequency picker (daily / weekdays / weekends / weekly /
    custom-days) + time → recurring
- **Cron (advanced)** mode keeps the raw cron text input plus the
  preset list, for power users.

The picker emits both `schedule_cron` and `one_shot`; the rest of the
form is unchanged. The reminder fieldset also exposes the `speaker`
channel and a **Personalize** checkbox (default on); see
[Personalization](#personalization).

The picker component lives at
`services/agent/frontend/src/routes/autonomy/ReminderTimePicker.svelte`.

## Internals worth knowing

- **The server is a thin REST wrapper.** All business logic — cron
  validation, scheduling, quiet-hours, delivery — lives in the autonomy
  engine. The MCP server only translates LLM-shaped tool calls into
  HTTP and back.
- **`in_seconds` rounds up to a minute boundary.** Cron resolution is
  one minute, so a reminder requested for "in 30 seconds" is scheduled
  for the start of the next minute. This is intentional — undershoot
  would cause the reminder to fire too soon (during the same minute the
  user already passed).
- **Recurring reminders are deletes-only on cancel.** There is no
  edit-in-place tool; to change a recurring reminder, cancel it and
  schedule a new one.
- **Filtering happens client-side.** `GET /autonomy/items` returns all
  agenda kinds; `list_reminders` filters to `kind='reminder'` (and to
  `enabled=true` unless `include_disabled` is set) inside the tool.

## Troubleshooting

### `autonomy api returned 404` on schedule

The autonomy router is mounted under `/api`, so the full path is
`/api/autonomy/items`. If a future agent refactor moves it, update
`AUTONOMY_ITEMS_URL` in `mcp_server.py`.

### `autonomy api returned 405: Method Not Allowed`

Same root cause as 404 — historically this was the symptom when the
tool hit `/autonomy/items` instead of `/api/autonomy/items`. Confirm:

```bash
docker compose exec agent curl -I http://localhost:6002/api/autonomy/items
```

A 200 means the path is correct.

### Reminder schedules but never fires

Check the autonomy engine isn't paused or rate-limited:

```bash
curl -s http://localhost/api/autonomy/status | jq
docker compose logs --tail 100 agent | grep -i autonomy
```

Quiet hours can also defer or drop a fire — see
[autonomy/v3.md → Quiet hours](../autonomy/v3.md). Manual triggers via
`POST /api/autonomy/trigger/{id}?bypass_quiet=true` will fire even
inside quiet hours.

### Cron syntax errors

The autonomy API validates cron strings via `croniter` before insert.
Errors come back as `400` with `invalid cron: <reason>`. The five
fields are `minute hour day-of-month month day-of-week`. Examples:

- `"0 18 * * 0"` — Sundays at 6pm
- `"30 7 * * 1-5"` — weekdays at 7:30am
- `"*/15 9-17 * * 1-5"` — every 15 minutes, 9am–5pm, weekdays

## Related files

- `services/agent/selene_agent/modules/mcp_reminder_tools/mcp_server.py`
  — implementation.
- `services/agent/selene_agent/autonomy/reminder_personalize.py` —
  the one-shot LLM rewrite helper.
- `services/agent/selene_agent/autonomy/handlers/reminder.py` — fire-time
  delivery handler that the engine invokes for `kind='reminder'` items;
  invokes the personalize helper and `generate_image` for Signal.
- `services/agent/selene_agent/autonomy/engine.py` — `_fire_item` honors
  the `_delete_after_run` result flag for one-shot cleanup.
- `services/agent/selene_agent/autonomy/notifiers.py` — Signal /
  HA-push / speaker notifier implementations. `SignalNotifier.send`
  accepts `attachments`.
- `services/agent/selene_agent/api/autonomy.py` — REST endpoints this
  tool calls.
- `services/agent/frontend/src/routes/autonomy/AgendaForm.svelte` —
  reminder fieldset (channel, personalize, one-shot).
- `services/agent/frontend/src/routes/autonomy/ReminderTimePicker.svelte`
  — Easy/Cron schedule picker for reminder kind.
- `services/agent/tests/test_reminder_tool.py` — tool tests.
- `services/agent/tests/test_reminder_handler.py` — handler tests.
- `services/agent/tests/test_reminder_personalize.py` — personalize
  helper tests.

## See also

- [Autonomy v3](../autonomy/v3.md) — full `reminder` config schema and
  the kind-by-kind config reference.
- [Autonomy README](../autonomy/README.md) — engine architecture,
  notifier protocol, agenda data model.
- [General tools](general.md) — `send_signal_message`, the underlying
  Signal delivery used by `channel='signal'`.
