"""
Conversations API router — browse and retrieve stored conversation histories.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from selene_agent.utils.conversation_db import conversation_db
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.session_pool import SessionOrchestratorPool

logger = custom_logger.get_logger('loki')

router = APIRouter()


SUMMARY_PREFIX = "[Prior conversation summary]"


def _filter_messages_for_resume(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip the leading base system prompt before sending the post-hydrate
    messages to the dashboard. The `[Prior conversation summary]` system
    message stays — the UI renders it as a `summary` role card so the user
    sees what the model sees.
    """
    if not messages:
        return []
    first = messages[0]
    content = first.get("content")
    if (
        first.get("role") == "system"
        and isinstance(content, str)
        and not content.startswith(SUMMARY_PREFIX)
    ):
        return list(messages[1:])
    return list(messages)


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


@router.delete("/conversations")
async def delete_all_conversations():
    """Delete every stored conversation flush.

    Irreversible. Live pool sessions are untouched — their next flush will
    create fresh rows. The dashboard exposes this through a "Delete all"
    button on `/history` that requires an explicit confirmation prompt.
    """
    deleted = await conversation_db.delete_all_conversations()
    if deleted is None:
        raise HTTPException(status_code=500, detail="Failed to delete conversations")
    return {"deleted": deleted}


@router.delete("/conversations/{session_id}")
async def delete_conversation(
    session_id: str,
    id: int = Query(..., description="The flush row id to delete"),
):
    """Delete a single stored flush row.

    The `(session_id, id)` pair is required and must match — a mismatched pair
    deletes nothing and 404s. Granularity is per-flush because /history lists
    one row per flush; bulk session deletion is not exposed here. If the
    session is currently live in the pool, the in-memory orchestrator is
    untouched — the next flush will create a new row.
    """
    deleted = await conversation_db.delete_conversation_history(session_id, id)
    if deleted is None:
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": deleted}


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
        "messages": _filter_messages_for_resume(orch.messages or []),
    }
