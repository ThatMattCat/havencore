# Archive

Snapshots of completed work, kept for historical reference. **Not maintained.** The information here describes the state of the system at the time the document was written and may diverge from the current codebase.

For current documentation, start at [`docs/README.md`](../README.md).

## Contents

- **[agent-revamp-2026.md](agent-revamp-2026.md)** — Original design narrative for the agent service rewrite (FastAPI consolidation, SvelteKit dashboard, single-port deployment, OpenAI-compat surface). The architecture described here is the current state and is documented in [`docs/services/agent/README.md`](../services/agent/README.md); this file preserves the design rationale and migration notes.
- **[autonomy-v3.md](autonomy-v3.md)** — Original design doc for the user-programmable agenda (reactive triggers via MQTT/HA-webhook, quiet hours, per-item event rate limits, live WebSocket run feed, `/autonomy` dashboard). All described capabilities are now part of the engine — see [`docs/services/agent/autonomy/README.md`](../services/agent/autonomy/README.md).
- **[autonomy-v4.md](autonomy-v4.md)** — Original design doc for the speaker delivery channel, the `watch_llm` triage kind, and the supervised `act` tier with confirmation gating. All described capabilities are part of the engine — `act` ships flagged off (`AUTONOMY_ACT_ENABLED`).
- **plans/** — Pre-implementation planning docs.
- **specs/** — Detailed design specifications written before build-out.
