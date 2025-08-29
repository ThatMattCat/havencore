"""
Database operations for conversation history storage
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncpg
import uuid

logger = logging.getLogger(__name__)

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
        limit: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve conversation history from database"""
        if not self.pool:
            logger.error("Database pool not initialized")
            return None
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT conversation_data, created_at, metadata
                    FROM conversation_histories 
                    WHERE session_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    session_id,
                    limit
                )
            
            histories = []
            for row in rows:
                histories.append({
                    'messages': json.loads(row['conversation_data']),
                    'created_at': row['created_at'].isoformat(),
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                })
            
            return histories
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return None
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

# Global instance
conversation_db = ConversationHistoryDB()