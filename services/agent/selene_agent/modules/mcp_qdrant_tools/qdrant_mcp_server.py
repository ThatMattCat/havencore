"""
Qdrant MCP Server - A simple MCP server for vector database operations
Run with: python -m qdrant_mcp_server
"""

import os
import json
import uuid
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, DatetimeRange,
    Filter, FieldCondition, MatchValue, PayloadSchemaType
)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

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
        self.server = Server("qdrant-server")
        self._setup_handlers()
    
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

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from the embeddings service"""
        try:
            response = requests.post(
                f"{self.embeddings_url}/embed",
                json={"inputs": text}
            )
            response.raise_for_status()
            # TEI returns nested list for batch processing
            return response.json()[0]
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

    def _setup_handlers(self):
        """Set up MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="create_memory",
                    description="Store information in the vector database for future retrieval",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The content to store in the database"
                            },
                            "importance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "Importance level (1=low, 5=critical)",
                                "default": 3
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags for categorization",
                                "default": []
                            },
                            "expires_in_days": {
                                "type": "integer",
                                "description": "Optional expiry time in days"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="search_memories",
                    description="Search information stored in the vector database using semantic similarity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant information"
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "Maximum number of results to return",
                                "default": 5
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Optional: only search data from the last N days"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="delete_memory",
                    description=(
                        "Delete a stored memory by its id. Use this when the user asks "
                        "you to forget, delete, remove, or correct a stored fact. First "
                        "call `search_memories` to find the matching entry and read its "
                        "`id`; then call this tool with that id. Deletion is permanent."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "The id of the memory to delete (from search_memories results)."
                            }
                        },
                        "required": ["memory_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "create_memory":
                    result = await self._create_memory(arguments)
                elif name == "search_memories":
                    result = await self._search_memories(arguments)
                elif name == "delete_memory":
                    result = await self._delete_memory(arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            except Exception as e:
                logger.error(f"Error calling tool {name}: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )]
    
    async def _create_memory(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Store data in the vector database"""
        try:
            text = args["text"]
            importance = args.get("importance", 3)
            tags = args.get("tags", [])
            expires_in_days = args.get("expires_in_days")
            
            # Get embedding
            embedding = self._get_embedding(text)
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
            query_embedding = self._get_embedding(query)
            
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


    async def run(self):
        """Run the MCP server"""
        options = self.server.create_initialization_options()
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, options, raise_exceptions=True)


async def main():
    """Main entry point"""
    logger.info("Starting Qdrant MCP Server...")
    server = QdrantMCPServer()
    await server.run()


if __name__ == "__main__":
    # Only run if executed directly, not when imported
    import sys
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)