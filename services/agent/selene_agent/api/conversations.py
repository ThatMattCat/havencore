"""
Conversations API router — browse and retrieve stored conversation histories.
"""

from fastapi import APIRouter, HTTPException

from selene_agent.utils.conversation_db import conversation_db
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()


@router.get("/conversations")
async def list_conversations(limit: int = 20, offset: int = 0):
    """List recent conversations with pagination"""
    conversations = await conversation_db.list_conversations(limit=limit, offset=offset)
    if conversations is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")
    return {"conversations": conversations, "limit": limit, "offset": offset}


@router.get("/conversations/{session_id}")
async def get_conversation(session_id: str):
    """Get a specific conversation by session ID"""
    history = await conversation_db.get_conversation_history(session_id)
    if history is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")
    if not history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": history}
