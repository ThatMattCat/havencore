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
    Filter, FieldCondition
)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('qdrant_mcp')

# Try to import config, fall back to defaults if not available
try:
    # Try relative import first (when run as module)
    from .config import (
        QDRANT_HOST, QDRANT_PORT, EMBEDDINGS_URL, 
        EMBEDDING_DIM, COLLECTION_NAME
    )
except (ImportError, ValueError):
    try:
        # Try absolute import (when run directly)
        from config import (
            QDRANT_HOST, QDRANT_PORT, EMBEDDINGS_URL, 
            EMBEDDING_DIM, COLLECTION_NAME
        )
    except ImportError:
        # Fall back to environment variables or defaults
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
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
    
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
    
    def _setup_handlers(self):
        """Set up MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="create",
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
                    name="search",
                    description="Search stored information using semantic similarity",
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
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "create":
                    result = await self._create_memory(arguments)
                elif name == "search":
                    result = await self._search_memories(arguments)
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
                "source": "mcp_server"
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
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            memories = []
            for result in results:
                memory = {
                    "id": str(result.id),
                    "text": result.payload.get("text", ""),
                    "timestamp": result.payload.get("timestamp", ""),
                    "importance": result.payload.get("importance", 0),
                    "tags": result.payload.get("tags", []),
                    "relevance_score": float(result.score)
                }
                
                # Include expiry info if present
                if "expires" in result.payload:
                    memory["expires"] = result.payload["expires"]
                
                memories.append(memory)
            
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