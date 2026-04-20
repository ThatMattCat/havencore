# TODO

Forward-looking items that aren't in-flight. Each bullet is a seed for a later pass; details to be fleshed out when the work is picked up.

## Chat / session lifecycle

- **Disable (or vastly raise) idle-summarize for dashboard chat sessions.** Today every session inherits `CONVERSATION_TIMEOUT` (default ~90s) unless the client passes `X-Idle-Timeout` / `idle_timeout`; after that window the idle sweep runs `_summarize_and_reset` and the thread is silently compressed. For an interactive dashboard tab this is the wrong default — the user expects a conversation to last until they hit "New Chat." Options: (a) dashboard sends a very large `idle_timeout` on every WS open frame, (b) add a sentinel value (`0` / `-1` / `null`) meaning "never auto-reset" and wire it through `effective_timeout()` and the sweep. Keep pucks/satellites on the current short window — they want summarization.

## Resume-from-history UX

- **Resume should repopulate `/chat` with the orchestrator's actual `messages`, not an empty pane.** Today the Resume button hydrates the session server-side and navigates to `/chat`, but the chat transcript doesn't visually reflect what the LLM will see on the next turn. After resume, the Chat pane should clear any stale transcript and render exactly the post-hydrate `messages` — i.e. `[system prompt + L4] + [Prior conversation summary] + tail exchanges`, with the base system prompt hidden and the summary shown in the same distinct styling we use on `/history`. The user should see what the model sees.

## Compaction visibility in the dashboard

- **Surface `session_summarize_reset` events in the chat UI in real time.** When the idle sweep compacts a live session the user currently sees nothing — the next reply just silently happens on a rewritten message list. The Chat pane should render an inline marker when a reset fires (e.g. "— conversation summarized —") and expose the stored `metadata.rolling_summary` so the user can click to see exactly what got folded into the summary. Tie-in: the prompt-cache log line `cache read=0 create=N` is a reliable signal that a compaction round just happened; the marker should show even for users not watching the agent logs.

## Dashboard session timeout isn't respected

- **Investigate why the dashboard's "idle timeout" setting seems shorter than configured.** At least one session hit `session_summarize_reset` well before the UI's configured timeout should have fired. Candidates: (a) the UI value isn't being sent on the WS open frame or gets overwritten by a later message without `idle_timeout`, (b) `effective_timeout()` falls back to `CONVERSATION_TIMEOUT` when the field is null/0, (c) the sweep runs against a stale orchestrator snapshot. Repro: set a long timeout in the UI, leave a session idle just under it, confirm no reset fires. This overlaps with the first TODO but deserves its own diagnosis pass first.

## History detail parity with what the LLM received

- **`/history` detail should mirror the LLM's view of each flush, not the raw pre-flush buffer.** When a session is summarized, the stored flush includes the pre-summary messages (captured for auditing) plus `metadata.rolling_summary`. The dashboard currently renders the pre-summary messages too, which misrepresents what context the LLM actually had on subsequent turns. Flip the default: for rows where `metadata.rolling_summary` is set, show the summary (and any post-reset exchanges, if we begin storing those as separate flushes) rather than the pre-reset transcript. Keep the raw pre-reset transcript accessible — maybe via a "show raw" toggle — since it's still useful for debugging.
