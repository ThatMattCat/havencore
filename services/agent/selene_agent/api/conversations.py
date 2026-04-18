"""
Conversations API router — browse and retrieve stored conversation histories.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from selene_agent.utils.conversation_db import conversation_db
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.session_pool import SessionOrchestratorPool

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
async def get_conversation(
    session_id: str,
    id: Optional[int] = Query(default=None),
):
    """Get stored conversation histories for a session.

    Without `id`, returns every stored flush for this session_id ordered
    newest-first. With `id`, returns only the single flush whose primary key
    matches (scoped to `session_id` so a mismatched pair 404s).
    """
    history = await conversation_db.get_conversation_history(session_id, flush_id=id)
    if history is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation")
    if not history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": history}


@router.post("/conversations/{session_id}/resume")
async def resume_conversation(session_id: str, req: Request):
    """Hydrate a stored session into the live pool so /chat can continue it.

    The pool either finds it already in memory, cold-resumes from
    conversation_db, or — if the session_id is unknown in both — falls through
    to minting a fresh session (in which case `resumed=false`).
    """
    pool: SessionOrchestratorPool = req.app.state.session_pool
    if not pool:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    orch = await pool.get_or_create(session_id)
    message_count = len(orch.messages) if orch.messages else 0
    # Messages > 1 means the system prompt plus at least one real turn
    # survived the hydrate path. A fresh mint would be exactly 1 (system only).
    resumed = orch.session_id == session_id and message_count > 1
    return {
        "session_id": orch.session_id,
        "resumed": resumed,
        "message_count": message_count,
    }
