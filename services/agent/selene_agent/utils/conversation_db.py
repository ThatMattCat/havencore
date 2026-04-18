"""
Database operations for conversation history storage
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncpg
import uuid

from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

class ConversationHistoryDB:
    """Database operations for storing conversation histories"""
    
    def __init__(self):
        self.pool = None
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'havencore'),
            'user': os.getenv('POSTGRES_USER', 'havencore'),
            'password': os.getenv('POSTGRES_PASSWORD', 'havencore_password')
        }
    
    async def initialize(self, max_retries: int = 10, retry_delay: int = 5):
        """Initialize database connection pool with retries"""
        for attempt in range(max_retries):
            try:
                self.pool = await asyncpg.create_pool(
                    **self.db_config,
                    min_size=1,
                    max_size=10,
                    command_timeout=60
                )
                logger.info("PostgreSQL connection pool initialized successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to connect to PostgreSQL (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Failed to initialize PostgreSQL connection after all retries")
                    raise
    
    async def store_conversation_history(
        self,
        messages: List[Dict[str, Any]], 
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store conversation history to database"""
        if not self.pool:
            logger.error("Database pool not initialized")
            return False
        
        try:
            if not session_id:
                session_id = str(uuid.uuid4())

            if not metadata:
                metadata = {}
            
            # Add timestamp to metadata
            metadata['stored_at'] = datetime.utcnow().isoformat()
            
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO conversation_histories (session_id, conversation_data, metadata)
                    VALUES ($1, $2, $3)
                    """,
                    session_id,
                    json.dumps(messages),
                    json.dumps(metadata)
                )
            
            logger.info(f"Stored conversation history with {len(messages)} messages for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store conversation history: {e}")
            return False
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 100,
        flush_id: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve conversation history from database.

        When `flush_id` is provided, returns at most one row (the specific flush
        with that primary key, scoped to the given session_id for safety).
        Otherwise returns up to `limit` rows for the session, newest first.
        """
        if not self.pool:
            logger.error("Database pool not initialized")
            return None

        try:
            async with self.pool.acquire() as conn:
                if flush_id is not None:
                    rows = await conn.fetch(
                        """
                        SELECT id, conversation_data, created_at, metadata
                        FROM conversation_histories
                        WHERE session_id = $1 AND id = $2
                        """,
                        session_id,
                        flush_id,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id, conversation_data, created_at, metadata
                        FROM conversation_histories
                        WHERE session_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                        """,
                        session_id,
                        limit,
                    )

            histories = []
            for row in rows:
                histories.append({
                    'id': row['id'],
                    'messages': json.loads(row['conversation_data']),
                    'created_at': row['created_at'].isoformat(),
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                })

            return histories

        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return None
    
    async def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """List recent conversations with pagination"""
        if not self.pool:
            logger.error("Database pool not initialized")
            return None

        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, created_at, metadata
                    FROM conversation_histories
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )

            conversations = []
            for row in rows:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                conversations.append({
                    'id': row['id'],
                    'session_id': row['session_id'],
                    'created_at': row['created_at'].isoformat(),
                    'message_count': metadata.get('message_count', 0),
                    'agent_name': metadata.get('agent_name', ''),
                    'metadata': metadata,
                })

            return conversations

        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")
            return None

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

# Global instance
conversation_db = ConversationHistoryDB()