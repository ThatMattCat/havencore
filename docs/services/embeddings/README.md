# Embeddings Service

HuggingFace Text Embeddings Inference (TEI) container. Produces vectors consumed by the [Qdrant](../qdrant/README.md) vector database for semantic memory.

## Purpose

- Convert text to vector embeddings
- Multiple embedding-model support
- Integration with the vector database
- Semantic similarity computation

## Configuration

```yaml
embeddings:
  image: ghcr.io/huggingface/text-embeddings-inference:1.8
  ports:
    - "3000:3000"
  environment:
    - MODEL_ID=BAAI/bge-large-en-v1.5
    - CUDA_VISIBLE_DEVICES=${EMBEDDINGS_GPU:-0}
```

The default model is `bge-large-en-v1.5` (1024-dim). Set `EMBEDDING_DIM` in the agent's env to match whatever model you run.

The host GPU comes from `EMBEDDINGS_GPU` in `.env` (default `0`). With the default chat model, GPUs 0-3 belong to the `vllm` service, so the reference host sets this to `4` — see [configuration.md → GPU Settings](../../configuration.md#gpu-settings).

## API usage

```bash
# Generate embeddings
curl -X POST http://localhost:3000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["Hello world", "Another sentence"]
  }'
```

## Supported models

- `BAAI/bge-large-en-v1.5` (1024 dimensions) — default
- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- `sentence-transformers/all-mpnet-base-v2` (768 dimensions)

## Related

- [Qdrant service](../qdrant/README.md) — stores the vectors this service produces.
- [MCP Qdrant tools](../agent/tools/qdrant.md) — agent-side tools that use this pipeline.
