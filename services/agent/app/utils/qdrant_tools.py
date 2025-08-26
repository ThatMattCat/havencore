"""
Qdrant Vector Database Tools for AI Assistant
Provides function-calling interface for storing and retrieving memories
"""

import json
import uuid
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, DatetimeRange, 
    Filter, FieldCondition, Range, MatchValue,
    SearchRequest, OrderBy, Direction
)
import config
import shared.scripts.logger as logger_module
logger = logger_module.get_logger('loki')


class QdrantTools:
    """Tools for AI assistant to interact with Qdrant vector database"""
    
    def __init__(self):
        """Initialize Qdrant client and ensure collections exist"""
        self.client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )
        self.embeddings_url = config.EMBEDDINGS_URL
        self.embedding_dim = config.EMBEDDING_DIM
        
        # Initialize collections if they don't exist
        self._init_collections()
    
    def _init_collections(self):
        """Create collections if they don't exist"""
        for collection_name in config.COLLECTION_NAMES:
            try:
                self.client.get_collection(collection_name)
            except:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {collection_name}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from the embeddings service"""
        response = requests.post(
            f"{self.embeddings_url}/embed",
            json={"inputs": text}
        )
        response.raise_for_status()
        # TEI returns nested list for batch processing
        return response.json()[0]
    
    def store_memory(
        self,
        text: str,
        category: str = "general",
        collection: str = "knowledge",
        importance: int = 3,
        tags: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Store a memory/fact/conversation in the vector database
        
        Args:
            text: The content to store
            category: Type of memory (conversation, fact, preference, task)
            collection: Which collection to use (conversations, knowledge)
            importance: Priority level 1-5
            tags: Optional list of tags for filtering
            expires_in_days: Optional expiry time in days
        
        Returns:
            Dictionary with memory_id and status
        """
        try:
            embedding = self._get_embedding(text)
            #print(f"Text embedding is: {embedding}")
            logger.debug(f"Generated embedding of length {len(embedding)} for text: \"{text}\"")

            memory_id = str(uuid.uuid4())
            
            payload = {
                "text": text,
                "category": category,
                "timestamp": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "importance": importance,
                "tags": tags or [],
                "source": "assistant"
            }
            
            if expires_in_days:
                expiry = (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).strftime('%Y-%m-%dT%H:%M:%SZ')
                payload["expires"] = expiry

            self.client.upsert(
                collection_name=collection,
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
                "collection": collection,
                "message": f"Memory stored successfully in {collection}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_memories(
        self,
        query: str,
        collection: str = "knowledge",
        category_filter: Optional[str] = None,
        days_back: Optional[int] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search for relevant memories using semantic similarity
        
        Args:
            query: Search query text
            collection: Which collection to search
            category_filter: Optional category to filter by
            days_back: Optional time range filter (search last N days)
            limit: Maximum number of results
        
        Returns:
            Dictionary with search results and relevance scores
        """
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Build filters
            filters = []

            # Filter out expired memories

            filters.append(
                FieldCondition(
                    key="expires",
                    range=DatetimeRange(
                        gt=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                    )
                )
            )

            if category_filter:
                filters.append(
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category_filter)
                    )
                )
            
            if days_back:
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime('%Y-%m-%dT%H:%M:%SZ')
                filters.append(
                    FieldCondition(
                        key="timestamp",
                        range=DatetimeRange(gte=cutoff_date)
                    )
                )
            
            # Search
            search_filter = Filter(should=filters) if filters else None
            
            results = self.client.search(
                collection_name=collection,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            memories = []
            for result in results:
                memory = {
                    "id": result.id,
                    "text": result.payload.get("text", ""),
                    "category": result.payload.get("category", ""),
                    "timestamp": result.payload.get("timestamp", ""),
                    "importance": result.payload.get("importance", 0),
                    "tags": result.payload.get("tags", []),
                    "relevance_score": result.score
                }
                memories.append(memory)
            
            return {
                "success": True,
                "query": query,
                "collection": collection,
                "count": len(memories),
                "memories": memories
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_memory(
        self,
        memory_id: str,
        collection: str = "knowledge",
        new_text: Optional[str] = None,
        new_category: Optional[str] = None,
        new_importance: Optional[int] = None,
        new_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Update an existing memory's content or metadata
        
        Args:
            memory_id: ID of the memory to update
            collection: Which collection the memory is in
            new_text: Updated text content (will re-embed)
            new_category: Updated category
            new_importance: Updated importance level
            new_tags: Updated tags list
        
        Returns:
            Dictionary with update status
        """
        try:
            # Get existing point
            existing = self.client.retrieve(
                collection_name=collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False
            )
            
            if not existing:
                return {
                    "success": False,
                    "error": f"Memory {memory_id} not found in {collection}"
                }
            
            payload = existing[0].payload
            
            # Update fields
            if new_category is not None:
                payload["category"] = new_category
            if new_importance is not None:
                payload["importance"] = new_importance
            if new_tags is not None:
                payload["tags"] = new_tags
            
            # Handle text update (requires new embedding)
            if new_text:
                payload["text"] = new_text
                payload["updated_at"] = datetime.now().isoformat()
                embedding = self._get_embedding(new_text)
                
                # Update with new vector
                self.client.upsert(
                    collection_name=collection,
                    points=[
                        PointStruct(
                            id=memory_id,
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )
            else:
                # Update payload only
                self.client.set_payload(
                    collection_name=collection,
                    payload=payload,
                    points=[memory_id]
                )
            
            return {
                "success": True,
                "memory_id": memory_id,
                "collection": collection,
                "message": "Memory updated successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_recent(
        self,
        collection: str = "knowledge",
        category_filter: Optional[str] = None,
        limit: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List recent memories chronologically
        
        Args:
            collection: Which collection to query
            category_filter: Optional category to filter by
            limit: Maximum number of results
            offset: Pagination offset
        
        Returns:
            Dictionary with recent memories
        """
        try:
            # Build filter
            filters = []
            if category_filter:
                filters.append(
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category_filter)
                    )
                )
            
            search_filter = Filter(must=filters) if filters else None
            
            # Scroll through points (no vector search, just retrieval)
            results, next_offset = self.client.scroll(
                collection_name=collection,
                scroll_filter=search_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            # Sort by timestamp (most recent first)
            memories = []
            for point in results:
                memory = {
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "category": point.payload.get("category", ""),
                    "timestamp": point.payload.get("timestamp", ""),
                    "importance": point.payload.get("importance", 0),
                    "tags": point.payload.get("tags", [])
                }
                memories.append(memory)
            
            # Sort by timestamp
            memories.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return {
                "success": True,
                "collection": collection,
                "count": len(memories),
                "has_more": next_offset is not None,
                "next_offset": next_offset,
                "memories": memories
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_memory(
        self,
        memory_id: Optional[str] = None,
        collection: str = "knowledge",
        category_filter: Optional[str] = None,
        older_than_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Delete a specific memory or memories matching criteria
        
        Args:
            memory_id: Specific memory ID to delete
            collection: Which collection to delete from
            category_filter: Delete all in category
            older_than_days: Delete memories older than N days
        
        Returns:
            Dictionary with deletion status
        """
        try:
            if memory_id:
                # Delete specific memory
                self.client.delete(
                    collection_name=collection,
                    points_selector=[memory_id]
                )
                return {
                    "success": True,
                    "message": f"Deleted memory {memory_id}",
                    "deleted_count": 1
                }
            
            # Build filter for bulk deletion
            filters = []
            if category_filter:
                filters.append(
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category_filter)
                    )
                )
            
            if older_than_days:
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).strftime('%Y-%m-%dT%H:%M:%SZ')
                filters.append(
                    FieldCondition(
                        key="timestamp",
                        range=DatetimeRange(lt=cutoff_date)
                    )
                )
            
            if not filters:
                return {
                    "success": False,
                    "error": "Must specify memory_id or filter criteria"
                }
            
            # Count before deletion
            count_filter = Filter(must=filters)
            count = self.client.count(
                collection_name=collection,
                count_filter=count_filter
            ).count
            
            # Delete matching points
            self.client.delete(
                collection_name=collection,
                points_selector=count_filter
            )
            
            return {
                "success": True,
                "message": f"Deleted {count} memories",
                "deleted_count": count
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def get_tool_definitions() -> List[Dict[str, Any]]:
        """Return OpenAI-compatible tool definitions"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "store_memory",
                    "description": "Store a memory, fact, conversation snippet, or information about the user or their environment in the vector database for future retrieval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The content to store (fact, conversation, preference, etc.)"
                            },
                            "category": {
                                "type": "string",
                                "enum": ["conversation", "fact", "preference", "task", "general"],
                                "description": "Type of memory being stored"
                            },
                            "collection": {
                                "type": "string",
                                "enum": ["conversations", "knowledge"],
                                "description": "Which collection to store in (conversations for chat history, knowledge for facts/info)"
                            },
                            "importance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "Importance level (1=low, 5=critical)"
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional tags for categorization"
                            },
                            "expires_in_days": {
                                "type": "integer",
                                "description": "Optional expiry time in days (for temporary information)"
                            }
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_memories",
                    "description": "Search stored memories using semantic similarity to find relevant information, past conversations, or facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant memories"
                            },
                            "collection": {
                                "type": "string",
                                "enum": ["conversations", "knowledge"],
                                "description": "Which collection to search"
                            },
                            "category_filter": {
                                "type": "string",
                                "enum": ["conversation", "fact", "preference", "task", "general"],
                                "description": "Optional: filter results by category"
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Optional: only search memories from the last N days"
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "Maximum number of results to return"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_memory",
                    "description": "Update an existing memory's content or metadata (category, importance, tags).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the memory to update"
                            },
                            "collection": {
                                "type": "string",
                                "enum": ["conversations", "knowledge"],
                                "description": "Which collection the memory is in"
                            },
                            "new_text": {
                                "type": "string",
                                "description": "Updated text content (will re-embed)"
                            },
                            "new_category": {
                                "type": "string",
                                "enum": ["conversation", "fact", "preference", "task", "general"],
                                "description": "Updated category"
                            },
                            "new_importance": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "Updated importance level"
                            },
                            "new_tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Updated tags list"
                            }
                        },
                        "required": ["memory_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_recent",
                    "description": "List recent memories chronologically, useful for context awareness and reviewing recent interactions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "collection": {
                                "type": "string",
                                "enum": ["conversations", "knowledge"],
                                "description": "Which collection to query"
                            },
                            "category_filter": {
                                "type": "string",
                                "enum": ["conversation", "fact", "preference", "task", "general"],
                                "description": "Optional: filter by category"
                            },
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 50,
                                "description": "Maximum number of results"
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Pagination offset"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_memory",
                    "description": "Delete a specific memory or memories matching criteria (outdated, incorrect, or expired information).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "Specific memory ID to delete"
                            },
                            "collection": {
                                "type": "string",
                                "enum": ["conversations", "knowledge"],
                                "description": "Which collection to delete from"
                            },
                            "category_filter": {
                                "type": "string",
                                "enum": ["conversation", "fact", "preference", "task", "general"],
                                "description": "Delete all memories in this category"
                            },
                            "older_than_days": {
                                "type": "integer",
                                "description": "Delete memories older than N days"
                            }
                        },
                        "required": []
                    }
                }
            }
        ]


# Example usage
if __name__ == "__main__":
    tools = QdrantTools()
    
    # Get tool definitions for OpenAI
    definitions = tools.get_tool_definitions()
    print(json.dumps(definitions[0], indent=2))

    # Test storing a memory
    result = tools.store_memory(
        text="User prefers dark mode for all interfaces",
        category="preference",
        collection="knowledge",
        importance=4,
        tags=["ui", "settings"]
    )
    query_result = tools.search_memories(
        query="What are the user's interface preferences?",
        collection="knowledge",
    )

    print(f"\nStore result: {result}")