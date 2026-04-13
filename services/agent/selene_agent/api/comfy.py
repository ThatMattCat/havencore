"""ComfyUI proxy — fire-and-poll image generation for the dashboard."""
import asyncio
import time
import uuid
from typing import Any, Dict, Optional

import aiohttp
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from selene_agent.modules.mcp_general_tools.comfyui_tools import SimpleComfyUI
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()

COMFY_BASE = "http://text-to-image:8188"

_jobs: Dict[str, Dict[str, Any]] = {}


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    workflow: Optional[str] = "default"


async def _run_job(job_id: str, req: GenerateRequest) -> None:
    job = _jobs[job_id]
    try:
        async with SimpleComfyUI("text-to-image:8188") as comfy:
            kwargs: Dict[str, Dict[str, Any]] = {}
            if req.steps is not None:
                kwargs["3"] = {"steps": req.steps}
            result = await comfy.text_to_image(
                prompt=req.prompt,
                workflow_name=req.workflow or "default",
                negative=req.negative_prompt,
                seed=req.seed,
                **kwargs,
            )
        images = [
            {"filename": img["filename"], "url": f"/outputs/{img['filename']}"}
            for img in result.get("images", [])
        ]
        job["status"] = "done"
        job["images"] = images
        job["prompt_id"] = result.get("prompt_id")
        job["finished_at"] = time.time()
    except Exception as e:
        logger.error(f"ComfyUI job {job_id} failed: {e}")
        job["status"] = "error"
        job["error"] = str(e)
        job["finished_at"] = time.time()


@router.post("/comfy/generate")
async def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "started_at": time.time(),
        "images": [],
    }
    asyncio.create_task(_run_job(job_id, req))
    return {"job_id": job_id, "status": "pending"}


@router.get("/comfy/status/{job_id}")
async def status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    elapsed_ms = int(((job.get("finished_at") or time.time()) - job["started_at"]) * 1000)
    return {
        "job_id": job_id,
        "status": job["status"],
        "elapsed_ms": elapsed_ms,
        "images": job.get("images", []),
        "error": job.get("error"),
    }


@router.get("/comfy/health")
async def health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{COMFY_BASE}/system_stats",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status >= 400:
                    return JSONResponse(status_code=resp.status, content={"status": "unhealthy"})
                return {"status": "healthy"}
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "unhealthy", "error": str(e)})
