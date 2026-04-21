# TODO

Forward-looking items that aren't in-flight. Each bullet is a seed for a later pass; details to be fleshed out when the work is picked up.

## Resume-from-history UX

- **Resume should repopulate `/chat` with the orchestrator's actual `messages`, not an empty pane.** Today the Resume button hydrates the session server-side and navigates to `/chat`, but the chat transcript doesn't visually reflect what the LLM will see on the next turn. After resume, the Chat pane should clear any stale transcript and render exactly the post-hydrate `messages` — i.e. `[system prompt + L4] + [Prior conversation summary] + tail exchanges`, with the base system prompt hidden and the summary shown in the same distinct styling we use on `/history`. The user should see what the model sees.

## Context-size-triggered summarization

- **Auto-summarize any session when its context size crosses a threshold.** With the dashboard's `idle_timeout=-1` sentinel, dashboard sessions now never auto-reset on idle — they live until "New Chat" or LRU eviction, and their message list grows unbounded. Pucks/satellites also risk this if a single exchange balloons (long tool outputs, big multimodal payloads). Add a size-based trigger parallel to the idle one: when total token/char count of `messages` exceeds some limit (TBD — probably a fraction of the model's context window, e.g. 75%), fire `_summarize_and_reset(reason="context_size_summarize")`. Hook points: likely in `AgentOrchestrator.run()` alongside the existing `_check_session_timeout()` call (`orchestrator.py:411`), and/or as a second gate in the pool sweep. Needs: (a) a cheap token-count estimator (reuse whatever the metrics path uses, or a char/4 approximation), (b) a new config var `CONVERSATION_CONTEXT_LIMIT_TOKENS` or similar, (c) a way to bypass/override for dashboard sessions if the user wants truly unbounded, or to set a higher ceiling there. Keep the `idle_timeout=-1` sentinel path unaffected — this is a separate axis.

## History detail parity with what the LLM received

- **`/history` detail should mirror the LLM's view of each flush, not the raw pre-flush buffer.** When a session is summarized, the stored flush includes the pre-summary messages (captured for auditing) plus `metadata.rolling_summary`. The dashboard currently renders the pre-summary messages too, which misrepresents what context the LLM actually had on subsequent turns. Flip the default: for rows where `metadata.rolling_summary` is set, show the summary (and any post-reset exchanges, if we begin storing those as separate flushes) rather than the pre-reset transcript. Keep the raw pre-reset transcript accessible — maybe via a "show raw" toggle — since it's still useful for debugging.

## Broken tests on main

- **`tests/test_memory_review_handler.py::test_prune_respects_source_protection` asserts `stats["l2_pruned"] == 1` but gets `2`.** The `_prune_l2` flow is deleting the protected L2 point even though an L3 still references it via `source_ids=["protected"]`. Either the protection lookup isn't being consulted or the L3 scroll filter is missing in the code path the test exercises. Pre-dates the LLM-provider branch (confirmed via `git stash`). Fix the handler or update the test to match the intended semantics — whichever is correct.
- **`tests/test_integration_memory_review.py::test_end_to_end_consolidation` fails with `requests.exceptions.MissingSchema: Invalid URL '/api/memory/runs/trigger'`.** The test uses bare `requests` with a relative path; it needs either a `base_url` or a FastAPI `TestClient`. Probably broke when the endpoint was moved and nobody re-ran this integration test. Also pre-dates the LLM-provider branch.

## LLM provider work — follow-ups

- **Wire the stubbed OpenAI provider.** `providers/openai.py` currently raises `NotImplementedError` and the factory falls back to vLLM when it's selected. Shape should mirror vLLM (`AsyncOpenAI` with a different base_url + key + model). The System-page button is already rendered disabled with "soon"; un-disable once the provider lands. Translation layer is not needed — OpenAI tool-calling is already OpenAI-shaped.
- **Consider caching the summarize-and-reset call too.** The `_summarize_session` path uses a different system prompt than normal chat so it's always a cache miss (`read=0 create=N`). Tiny cost per compaction, but trivial to add — one `cache_control` breakpoint on the summary's system block. Only worth doing if a profile shows compactions are fireing often enough to matter.

## Stretch goals

Post-MVP / "v2"-class features — multi-modal perception, identity awareness, richer output surfaces. To work on **after** the items above are cleared. Scoping, feasibility, effort, and suggested sequencing live in [`stretch-goals.md`](./stretch-goals.md).
