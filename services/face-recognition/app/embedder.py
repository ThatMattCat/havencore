"""InsightFace buffalo_l wrapper.

Loads RetinaFace (detection) + ArcFace R100 (512-d embedding) on the configured
GPU at startup. Subsequent calls reuse the cached FaceAnalysis app.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from insightface.app import FaceAnalysis

import config

logger = logging.getLogger("face-recognition.embedder")


@dataclass
class EmbedderInfo:
    model_pack: str
    ctx_id: int
    det_size: int
    providers: list[str]
    load_seconds: float


class Embedder:
    """Lazy-init wrapper around insightface.app.FaceAnalysis.

    `prepare()` is the expensive part (downloads + onnxruntime session init).
    Call once during FastAPI lifespan startup.
    """

    def __init__(self) -> None:
        self._app: FaceAnalysis | None = None
        self._info: EmbedderInfo | None = None

    def prepare(self) -> EmbedderInfo:
        if self._info is not None:
            return self._info

        t0 = time.monotonic()
        # CUDAExecutionProvider first; ORT will fall back to CPU only if CUDA
        # init fails, which we want to know about (logged).
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name=config.MODEL_PACK, providers=providers)
        app.prepare(ctx_id=config.CTX_ID, det_size=(config.DET_SIZE, config.DET_SIZE))

        active_providers = []
        # Each sub-model carries its own ORT session; sample the detector's.
        det = app.models.get("detection")
        if det is not None and hasattr(det, "session"):
            active_providers = list(det.session.get_providers())

        self._app = app
        self._info = EmbedderInfo(
            model_pack=config.MODEL_PACK,
            ctx_id=config.CTX_ID,
            det_size=config.DET_SIZE,
            providers=active_providers,
            load_seconds=round(time.monotonic() - t0, 2),
        )

        if "CUDAExecutionProvider" not in active_providers:
            logger.warning(
                "InsightFace loaded without CUDA — running on CPU. "
                "Active providers: %s",
                active_providers,
            )
        else:
            logger.info(
                "InsightFace %s loaded on CUDA in %.2fs (det_size=%d)",
                config.MODEL_PACK,
                self._info.load_seconds,
                config.DET_SIZE,
            )

        return self._info

    @property
    def info(self) -> EmbedderInfo | None:
        return self._info

    @property
    def ready(self) -> bool:
        return self._app is not None

    def detect_and_embed(self, bgr_image: np.ndarray):
        """Run detection + embedding on a single BGR image.

        Returns the list of insightface Face objects, each carrying:
        - `.bbox` (x1,y1,x2,y2)
        - `.kps` (5-point landmarks)
        - `.det_score`
        - `.normed_embedding` (512-d, L2-normalized)
        """
        if self._app is None:
            raise RuntimeError("Embedder.prepare() has not been called")
        return self._app.get(bgr_image)


embedder = Embedder()
