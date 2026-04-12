# IAV-to-Text Service

Image / audio / video understanding via a vision LLM. Accessed by the agent's MCP `query_multimodal_api` tool and by the dashboard's Vision playground.

## Status

Documentation stub — the service is shipping but doesn't yet have a dedicated doc. Add details here as the service stabilizes.

## Ports

- `8100` — primary API
- `8110` — secondary

## Where to look in the meantime

- `services/iav-to-text/` — service source, Dockerfile, and README (if any)
- `compose.yaml` — runtime configuration
- [MCP General Tools → `query_multimodal_api`](../agent/tools/general.md) — how the agent calls this service
- [Agent Service](../agent/README.md) — the `/api/vision/*` proxy endpoints for the dashboard playground
