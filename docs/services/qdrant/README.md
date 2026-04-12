# Qdrant Vector Database

Vector storage for semantic memory and retrieval-augmented use cases. Paired with the [Embeddings](../embeddings/README.md) service.

## Purpose

- Vector storage for embeddings
- Semantic search
- Retrieval-augmented generation (RAG)
- Document similarity matching

## Configuration

```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"  # HTTP API
    - "6334:6334"  # gRPC API
  volumes:
    - qdrant_data:/qdrant/storage
```

## API usage

```bash
# Create collection
curl -X PUT http://localhost:6333/collections/documents \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 384,
      "distance": "Cosine"
    }
  }'

# Add vectors
curl -X PUT http://localhost:6333/collections/documents/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {
        "id": 1,
        "vector": [0.1, 0.2, 0.3, ...],
        "payload": {"text": "Document content"}
      }
    ]
  }'

# Search similar vectors
curl -X POST http://localhost:6333/collections/documents/points/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, ...],
    "limit": 5
  }'
```

## Related

- [MCP Qdrant tools](../agent/tools/qdrant.md) — how the agent reads/writes semantic memory.
- [Embeddings service](../embeddings/README.md) — generates the vectors Qdrant stores.
