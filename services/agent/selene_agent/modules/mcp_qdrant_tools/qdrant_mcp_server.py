"""
Qdrant MCP Server - A simple MCP server for vector database operations
Run with: python -m selene_agent.modules.mcp_qdrant_tools

MCP surface is the mcp 2.0 ``MCPServer`` decorator API: the tools are typed
functions registered in ``QdrantMCPServer._build_mcp`` (see
``mcp_reminder_tools/mcp_server.py`` for the pattern). The Qdrant/embeddings
client layer and the module-level constants (``QDRANT_HOST``, ``QDRANT_PORT``,
``COLLECTION_NAME`` — imported by l4_context / retrieval / api.memory /
memory_review) are unchanged.
"""

import os
import json
import uuid
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Awaitable, Callable, Dict, List

import requests
from pydantic import Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, DatetimeRange,
    Filter, FieldCondition, MatchValue, PayloadSchemaType
)

from mcp.server import MCPServer

from selene_agent.modules._mcp_params import NULL_OK
from selene_agent.utils.logger import get_logger

logger = get_logger('loki')

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL", "http://embeddings:3000")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "user_data")


class QdrantMCPServer:
    """MCP Server for Qdrant vector database operations"""
    
    def __init__(self):
        """Initialize Qdrant client and ensure collection exists"""
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT
        )
        self.embeddings_url = EMBEDDINGS_URL
        self.embedding_dim = EMBEDDING_DIM
        self.collection_name = COLLECTION_NAME
        
        # Initialize collection if it doesn't exist
        self._init_collection()

        # Create MCP server
        self.mcp = self._build_mcp()
    
    def _init_collection(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
        self._init_payload_indexes()

    def _init_payload_indexes(self) -> None:
        """Idempotently create payload indexes required for v2 scroll/filter queries."""
        indexes = [
            ("tier", PayloadSchemaType.KEYWORD),
            ("pending_l4_approval", PayloadSchemaType.BOOL),
            ("importance_effective", PayloadSchemaType.FLOAT),
        ]
        for field_name, schema in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
                logger.info(f"Payload index created or already existed: {field_name}")
            except Exception as e:
                # Qdrant returns an error on re-create; log and continue.
                logger.debug(f"Payload index {field_name}: {e}")

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from the embeddings service.

        Offloaded to a worker thread with a bounded (connect, read) timeout: a
        wedged TEI (socket accepts but never answers) previously blocked this
        module's event loop forever, so every later create/search memory call
        also hung until the container restarted. The timeout fails fast and the
        thread offload keeps the loop responsive during a slow embed.
        """
        def _post() -> List[float]:
            response = requests.post(
                f"{self.embeddings_url}/embed",
                json={"inputs": text},
                timeout=(5, 30),
            )
            response.raise_for_status()
            # TEI returns nested list for batch processing
            return response.json()[0]

        try:
            return await asyncio.to_thread(_post)
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    async def _record_accesses(self, ids: List[str]) -> None:
        """Fire-and-forget bump of access_count + last_accessed_at for the given ids.

        Increment is approximate: Qdrant's set_payload is not atomic-increment.
        We read current counts and write back count+1. Concurrent retrievals may
        drop ticks — acceptable because consolidation applies log(1+access_count)
        which dampens counting noise.
        """
        if not ids:
            return
        try:
            current = self.client.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )
            now_iso = datetime.now(timezone.utc).isoformat()
            by_id = {str(p.id): (p.payload or {}).get("access_count", 0) for p in current}
            # Group ids by their new count so we can issue one set_payload per group.
            from collections import defaultdict
            groups = defaultdict(list)
            for pid in ids:
                groups[by_id.get(pid, 0) + 1].append(pid)
            for new_count, group_ids in groups.items():
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={
                        "access_count": new_count,
                        "last_accessed_at": now_iso,
                    },
                    points=group_ids,
                )
        except Exception as e:
            logger.warning(f"_record_accesses failed (non-fatal): {e}")

    def _build_mcp(self) -> MCPServer:
        """Register the decorated tool surface.

        structured_output=False on every tool: results stay a single
        TextContent JSON string, byte-identical to the pre-MCPServer wire
        format (no outputSchema in tools/list, no structuredContent).

        Note the metadata order on range-constrained optionals: ``Field(ge=
        ..., le=...)`` must come BEFORE ``NULL_OK``, or pydantic can no longer
        map the constraints onto the core int schema and emits literal
        ``"ge"``/``"le"`` keys instead of ``"minimum"``/``"maximum"``.
        """
        mcp = MCPServer("qdrant-server", version="1.0.0")

        @mcp.tool(
            name="create_memory",
            description="Store information in the vector database for future retrieval",
            structured_output=False,
        )
        async def create_memory(
            text: Annotated[str, Field(description="The content to store in the database")],
            importance: Annotated[int, Field(
                description="Importance level (1=low, 5=critical)", ge=1, le=5,
            ), NULL_OK] = 3,
            tags: Annotated[list[str], NULL_OK, Field(
                description="Optional tags for categorization",
            )] = [],
            expires_in_days: Annotated[int, NULL_OK, Field(
                description="Optional expiry time in days",
            )] = None,
        ) -> str:
            return await self._dispatch("create_memory", self._create_memory, {
                "text": text,
                "importance": importance,
                "tags": tags,
                "expires_in_days": expires_in_days,
            })

        @mcp.tool(
            name="search_memories",
            description="Search information stored in the vector database using semantic similarity",
            structured_output=False,
        )
        async def search_memories(
            query: Annotated[str, Field(description="Search query to find relevant information")],
            limit: Annotated[int, Field(
                description="Maximum number of results to return", ge=1, le=20,
            ), NULL_OK] = 5,
            days_back: Annotated[int, NULL_OK, Field(
                description="Optional: only search data from the last N days",
            )] = None,
        ) -> str:
            return await self._dispatch("search_memories", self._search_memories, {
                "query": query,
                "limit": limit,
                "days_back": days_back,
            })

        @mcp.tool(
            name="delete_memory",
            description=(
                "Delete a stored memory by its id. Use this when the user asks "
                "you to forget, delete, remove, or correct a stored fact. First "
                "call `search_memories` to find the matching entry and read its "
                "`id`; then call this tool with that id. Deletion is permanent."
            ),
            structured_output=False,
        )
        async def delete_memory(
            memory_id: Annotated[str, Field(
                description="The id of the memory to delete (from search_memories results).",
            )],
        ) -> str:
            return await self._dispatch("delete_memory", self._delete_memory, {
                "memory_id": memory_id,
            })

        return mcp

    async def _dispatch(
        self,
        name: str,
        impl: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        args: Dict[str, Any],
    ) -> str:
        """Run one tool impl with the old call_tool handler's exact contract.

        Indented-JSON text out; an unexpected exception becomes an
        ``{"error": ...}`` payload in ordinary (non-``isError``) content, so
        the agent-visible text stays identical to the hand-dispatch era.
        """
        try:
            result = await impl(args)
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            result = {"error": str(e)}
        return json.dumps(result, indent=2)
    
    async def _create_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store data in the vector database"""
        try:
            text = args["text"]
            importance = args.get("importance", 3)
            tags = args.get("tags", [])
            expires_in_days = args.get("expires_in_days")
            
            # Get embedding
            embedding = await self._get_embedding(text)
            logger.debug(f"Generated embedding of length {len(embedding)}")
            
            memory_id = str(uuid.uuid4())
            
            # Prepare payload
            payload = {
                "text": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "importance": importance,
                "tags": tags,
                "source": "mcp_server",
                # Memory tiering: new rows are L2. source_ids links consolidated
                # (L3/L4) entries back to originating L2 rows.
                "tier": "L2",
                "source_ids": [],
                # v2 access tracking + importance dynamics.
                "access_count": 0,
                "last_accessed_at": None,
                "importance_effective": importance,
                # v2 L4 proposal queue.
                "pending_l4_approval": False,
                "proposed_at": None,
                "proposal_rationale": None,
            }
            
            if expires_in_days:
                expiry = (datetime.now(timezone.utc) + timedelta(days=expires_in_days))
                payload["expires"] = expiry.isoformat()
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Successfully stored in {self.collection_name}",
                "timestamp": payload["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Failed to create memory: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _search_memories(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search for relevant memories using semantic similarity"""
        try:
            query = args["query"]
            limit = args.get("limit", 5)
            days_back = args.get("days_back")
            
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Build filters
            must_conditions = []
            must_not_conditions = []
            
            # Filter OUT expired memories (only if they have an expires field)
            # This uses must_not to exclude items where expires exists AND is in the past
            must_not_conditions.append(
                FieldCondition(
                    key="expires",
                    range=DatetimeRange(
                        lte=datetime.now(timezone.utc).isoformat()
                    )
                )
            )

            # v2: L4 entries are injected into every system prompt already — exclude
            # them from semantic retrieval to avoid wasting token budget.
            must_not_conditions.append(
                FieldCondition(key="tier", match=MatchValue(value="L4"))
            )

            # Filter by time range if specified
            if days_back:
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back))
                must_conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=DatetimeRange(gte=cutoff_date.isoformat())
                    )
                )
            
            # Build the filter
            search_filter = None
            if must_conditions or must_not_conditions:
                filter_dict = {}
                if must_conditions:
                    filter_dict["must"] = must_conditions
                if must_not_conditions:
                    filter_dict["must_not"] = must_not_conditions
                search_filter = Filter(**filter_dict)
            
            # Search in Qdrant
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=search_filter,
                limit=limit * 2,  # over-fetch slightly so tier re-ranking has room
                with_payload=True
            ).points

            from selene_agent.utils import config as cfg
            TIER_WEIGHT = {"L2": 1.0, "L3": cfg.MEMORY_L3_RANK_BOOST, "L4": 1.0}

            scored = []
            for result in results:
                tier = result.payload.get("tier", "L2")
                weight = TIER_WEIGHT.get(tier, 1.0)
                adjusted = float(result.score) * weight
                scored.append((adjusted, result))
            scored.sort(key=lambda t: t[0], reverse=True)
            scored = scored[:limit]

            memories = []
            for adjusted, result in scored:
                memory = {
                    "id": str(result.id),
                    "text": result.payload.get("text", ""),
                    "timestamp": result.payload.get("timestamp", ""),
                    "importance": result.payload.get("importance", 0),
                    "tags": result.payload.get("tags", []),
                    "tier": result.payload.get("tier", "L2"),
                    "source_ids": result.payload.get("source_ids", []),
                    "access_count": result.payload.get("access_count", 0),
                    "last_accessed_at": result.payload.get("last_accessed_at"),
                    "importance_effective": result.payload.get(
                        "importance_effective", result.payload.get("importance", 0)
                    ),
                    "relevance_score": float(result.score),
                    "adjusted_score": adjusted,
                }
                if "expires" in result.payload:
                    memory["expires"] = result.payload["expires"]
                memories.append(memory)

            # Fire-and-forget: do NOT await; retrieval must not wait on this.
            if memories:
                asyncio.create_task(self._record_accesses([m["id"] for m in memories]))

            return {
                "success": True,
                "query": query,
                "count": len(memories),
                "results": memories
            }

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return {
                "success": False,
                "error": str(e)
            }



    async def _delete_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a memory by id. Permanent; invalidates L4 cache when needed."""
        from qdrant_client.models import PointIdsList

        memory_id = args.get("memory_id")
        if not memory_id:
            return {"success": False, "error": "memory_id is required"}

        try:
            # Look up the memory first so we can report tier + invalidate L4 cache.
            existing = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
            if not existing:
                return {
                    "success": False,
                    "error": f"no memory with id {memory_id}",
                }
            tier = (existing[0].payload or {}).get("tier", "L2")

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[memory_id]),
            )

            if tier == "L4":
                try:
                    from selene_agent.utils import l4_context
                    l4_context.invalidate_cache()
                except Exception as e:
                    logger.warning(f"l4_context cache invalidate failed: {e}")

            return {
                "success": True,
                "memory_id": memory_id,
                "tier_deleted": tier,
            }
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return {"success": False, "error": str(e)}


def main():
    """Stdio entry point (``python -m selene_agent.modules.mcp_qdrant_tools``)."""
    logger.info("Starting Qdrant MCP Server (stdio)...")
    QdrantMCPServer().mcp.run("stdio")


if __name__ == "__main__":
    # Only run if executed directly, not when imported
    import sys
    try:
        main()  # MCPServer.run("stdio") is synchronous (it owns the event loop)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)