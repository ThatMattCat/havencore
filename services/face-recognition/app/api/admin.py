"""Admin / operator endpoints.

Trigger maintenance jobs out-of-band — sweeper for early eviction tests,
re-embedding (commit F) for drift recovery after a model swap.
"""

import logging

from fastapi import APIRouter

import retention


logger = logging.getLogger("face-recognition.api.admin")

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/retention/sweep")
async def trigger_retention_sweep():
    """Run a single retention sweep right now and return the result.

    Bypasses the periodic scheduler — useful when an operator just lowered
    a retention env var and wants the new policy applied without waiting
    for the next interval. The periodic loop continues unchanged.
    """
    result = await retention.run_once()
    retention.sweeper.last_result = result
    return result
