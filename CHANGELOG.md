# Changelog

All notable changes to HavenCore are tracked here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-04-16

First tagged release. HavenCore is a self-hosted, voice-capable AI smart-home
assistant running as a docker-compose stack on a Linux host with NVIDIA GPUs.
The v1 milestone covers the full agent loop (STT → LLM + tool calls → TTS),
a SvelteKit dashboard, semantic memory with nightly consolidation, an
autonomy engine that can speak and act with confirmations, and integrations
with Home Assistant, Plex, Music Assistant, and Signal.

### Added

- **Portfolio-framed README.** Rewritten as a tour of what the system does,
  the stack, and the notable engineering decisions behind it.
- **Dashboard chat: push-to-talk mic + auto-speak TTS.** The chat page now
  records straight from the browser, transcribes via the STT service, and
  speaks assistant turns back with Kokoro.
- **Kokoro native voices.** Voice playground and autonomy speaker exposes
  the full Kokoro voice set (not just the OpenAI-aliased subset).
- **Home Assistant entity suggestions.** When a tool call references an
  unknown entity_id, the agent now surfaces near-matches from the live HA
  entity registry instead of failing silently.
- **Autonomy v3 reactive triggers.** Webhook + MQTT event intake, quiet
  hours, per-item rate limits.
- **Autonomy v4 speak + act.** SpeakerNotifier runs Kokoro TTS through
  Music Assistant for room-scoped announcements; the `act` tier can run
  scripted changes behind explicit user confirmation.
- **Memory v2.** L1–L4 retrieval tiers with importance decay, nightly
  HDBSCAN clustering into L3 summaries, and an L4 persistent-memory block
  injected into the system prompt. Dashboard `/memory` page exposes
  browse, edit, proposal queue, and run history.
- **Music Assistant MCP module.** Audio-only playback routing to
  speakers, Chromecasts, and Google Homes, alongside the existing Plex
  video-oriented module.
- **Signal notifications.** Outbound messaging via `signal-cli-rest-api`
  for autonomy briefings, anomaly alerts, and act-confirmation links
  (replacing earlier email-based path).
- **Per-turn metrics persistence.** Orchestrator emits a METRIC event per
  turn; timings land in a `turn_metrics` Postgres table and render on
  the `/metrics` dashboard page.
- **Dashboard connection-state banner.** Dropped WebSocket connections
  surface a persistent reconnecting banner with a "Retry now" button
  instead of a silent red status dot.
- **Dashboard tool-failure card.** Assistant turns that end in an error
  render as a distinct error card instead of an empty bubble.

### Changed

- **Orchestrator is fully event-based.** THINKING / TOOL_CALL /
  TOOL_RESULT / METRIC / DONE / ERROR events flow through one generator,
  consumed by both the streaming WebSocket and the non-streaming REST
  endpoints.
- **Agent service is a single port.** Port 6002 now serves the SvelteKit
  SPA, `/api/*`, `/ws/*`, and the OpenAI-compatible `/v1/*` endpoints —
  no extra gateway needed for the dashboard.
- **MCP-first tool surface.** Tools are declared by MCP servers living
  under `services/agent/selene_agent/modules/` (`mcp_general_tools`,
  `mcp_homeassistant_tools`, `mcp_qdrant_tools`, `mcp_mqtt_tools`,
  `mcp_plex_tools`, `mcp_music_assistant_tools`). `UnifiedTool` abstracts
  both legacy and MCP sources through a single OpenAI function-calling
  schema.

### Fixed

- **MCP tool calls now have a hard timeout.** A wedged MCP server no
  longer blocks the orchestrator indefinitely; `MCP_TOOL_TIMEOUT_SECONDS`
  (default 120s) wraps every `call_tool()` with `asyncio.wait_for`, and a
  timeout returns a structured, recoverable error to the LLM.
- **Bare `except:` clauses replaced.** MCP schema parse and Wikipedia
  fallback now log typed exceptions with context.
- **Session-timeout resets are logged at INFO** with idle time,
  message count, and session id — previously silent at DEBUG.
- **Metrics activity bars render bottom-up** so recent activity reads
  naturally from left to right.
- **History pagination boundary.** "Next" correctly disables on the exact
  boundary where the last page has precisely `limit` items.
- **Documentation drift.** CLAUDE.md now lists the full MCP module set
  and points at the correct `frontend/` path; `.env.tmpl` uses
  consistent `python -m` invocations; MCP server tables mention Music
  Assistant.

[1.0.0]: https://github.com/ThatMattCat/havencore/releases/tag/v1.0.0
