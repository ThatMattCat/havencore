import base64
import io
import os
import subprocess
import tempfile
import time
import uuid
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch
from kokoro import KPipeline
import soundfile as sf

import config

if not config.SOLO:
    import shared.scripts.logger as logger_module
    logger = logger_module.get_logger('loki')
else:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('text-to-speech')

os.makedirs(config.AUDIO_DIR, exist_ok=True)

pipeline = KPipeline(lang_code=config.LANGUAGE, device=config.MODEL_DEVICE if torch.cuda.is_available() else 'cpu')


def _apply_pronunciation_overrides(pipe):
    """Inject custom pronunciations into misaki's Lexicon.

    Misaki has no inline phoneme syntax — the only way to force a word's
    pronunciation is to add it to the gold/silver dict consulted during G2P.
    Only English G2P (lang_code 'a'/'b') exposes a Lexicon; other languages
    are silently skipped.
    """
    overrides = getattr(config, "TTS_PRONUNCIATIONS", None)
    if not overrides:
        return
    lex = getattr(getattr(pipe, "g2p", None), "lexicon", None)
    if lex is None or not hasattr(lex, "golds"):
        logger.warning(
            "TTS_PRONUNCIATIONS configured but active G2P has no Lexicon "
            "(lang_code=%r); skipping",
            getattr(pipe, "lang_code", None),
        )
        return
    for word, phon in overrides.items():
        for variant in {word, word.lower(), word.capitalize(), word.upper()}:
            lex.golds[variant] = phon
        logger.info("TTS pronunciation override: %r -> %r", word, phon)


_apply_pronunciation_overrides(pipeline)

# Kokoro v1 voices, keyed by the language code the pipeline must run under.
# G2P is language-specific, so only voices whose prefix matches the configured
# LANGUAGE are exposed. Keep this in sync with hexgrad/Kokoro-82M on HF Hub.
KOKORO_VOICES_BY_LANG = {
    'a': [  # American English
        'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica',
        'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
        'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam',
        'am_michael', 'am_onyx', 'am_puck', 'am_santa',
    ],
    'b': [  # British English
        'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily',
        'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis',
    ],
    'j': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
    'z': [
        'zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
        'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang',
    ],
    'e': ['ef_dora', 'em_alex', 'em_santa'],
    'f': ['ff_siwis'],
    'h': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
    'i': ['if_sara', 'im_nicola'],
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],
}

AVAILABLE_VOICES = set(KOKORO_VOICES_BY_LANG.get(config.LANGUAGE, ['af_heart']))

# OpenAI-compat aliases resolve to the configured default voice so devices
# hard-wired to OpenAI's voice names still produce speech.
OPENAI_VOICE_ALIASES = {'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'}
DEFAULT_VOICE = config.VOICE if config.VOICE in AVAILABLE_VOICES else 'af_heart'

SAMPLE_RATE = 24000

# Rhubarb Lip Sync: produces a viseme timeline the companion app's
# Live2D overlay schedules against ExoPlayer playback position. Soft
# dependency — if the binary is missing or errors, the response just
# omits X-Visemes and the client falls back to closed-mouth playback.
RHUBARB_BIN = os.environ.get('RHUBARB_BIN', 'rhubarb')
RHUBARB_TIMEOUT_SEC = float(os.environ.get('RHUBARB_TIMEOUT_SEC', '10'))
RHUBARB_RECOGNIZER = os.environ.get('RHUBARB_RECOGNIZER', 'phonetic')

# Formats libsndfile can encode directly from raw samples. mp3/aac/pcm are
# handled separately (mp3/aac fall back to WAV since libsndfile can't
# produce them without extra codec libs).
_SF_ENCODERS = {
    'wav':  ('WAV',  'PCM_16', 'audio/wav'),
    'flac': ('FLAC', None,     'audio/flac'),
    'ogg':  ('OGG',  'VORBIS', 'audio/ogg'),
    'opus': ('OGG',  'OPUS',   'audio/ogg; codecs=opus'),
}


def encode_audio(samples: np.ndarray, fmt: str) -> tuple[bytes, str]:
    """Encode float32 PCM ``samples`` to ``fmt``. Returns ``(bytes, content_type)``.

    For formats libsndfile can't produce (mp3, aac), falls back to WAV with
    an honest ``audio/wav`` content-type rather than mislabeling the bytes.
    """
    fmt = (fmt or 'wav').lower()

    if fmt == 'pcm':
        pcm16 = np.clip(samples, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
        return pcm16, f'audio/L16; rate={SAMPLE_RATE}'

    spec = _SF_ENCODERS.get(fmt)
    if spec is None:
        logger.warning(
            f"Requested TTS format '{fmt}' is not supported by libsndfile; "
            "returning WAV with Content-Type: audio/wav"
        )
        spec = _SF_ENCODERS['wav']

    sf_format, subtype, content_type = spec
    buf = io.BytesIO()
    try:
        if subtype:
            sf.write(buf, samples, SAMPLE_RATE, format=sf_format, subtype=subtype)
        else:
            sf.write(buf, samples, SAMPLE_RATE, format=sf_format)
    except Exception as e:
        logger.warning(
            f"soundfile failed to encode as {sf_format}/{subtype} ({e}); "
            "falling back to WAV"
        )
        buf = io.BytesIO()
        sf.write(buf, samples, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        content_type = 'audio/wav'
    return buf.getvalue(), content_type


def compute_visemes(samples: np.ndarray) -> dict | None:
    """Run Rhubarb on ``samples`` and return the parsed viseme timeline.

    Returns ``None`` if Rhubarb is unavailable, times out, or errors —
    callers should treat the absence of an X-Visemes header as the
    soft-fallback signal (client renders silent mouth).
    """
    wav_fd, wav_path = tempfile.mkstemp(suffix='.wav', prefix='tts_visemes_')
    os.close(wav_fd)
    json_fd, json_path = tempfile.mkstemp(suffix='.json', prefix='tts_visemes_')
    os.close(json_fd)
    try:
        # Rhubarb requires 16-bit PCM WAV; libsndfile's float default is rejected.
        sf.write(wav_path, samples, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        result = subprocess.run(
            [RHUBARB_BIN, '-f', 'json', '-r', RHUBARB_RECOGNIZER,
             '-o', json_path, wav_path],
            capture_output=True,
            timeout=RHUBARB_TIMEOUT_SEC,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "rhubarb exited %d: %s",
                result.returncode,
                (result.stderr or '').strip()[:500],
            )
            return None
        with open(json_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(
            "rhubarb binary not found at %r; X-Visemes header disabled",
            RHUBARB_BIN,
        )
        return None
    except subprocess.TimeoutExpired:
        logger.warning(
            "rhubarb timed out after %.1fs; skipping visemes",
            RHUBARB_TIMEOUT_SEC,
        )
        return None
    except Exception as e:
        logger.warning("rhubarb failed: %s", e)
        return None
    finally:
        for p in (wav_path, json_path):
            try:
                os.unlink(p)
            except OSError:
                pass


class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/audio/speech":
            self.handle_speech_request()
        else:
            self.send_error_response(404, "Not found")

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        elif self.path == "/v1/voices":
            self.handle_voices_request()
        else:
            self.send_error_response(404, "Not found")

    def handle_voices_request(self):
        body = {
            "language": config.LANGUAGE,
            "default": DEFAULT_VOICE,
            "native": sorted(AVAILABLE_VOICES),
            "aliases": sorted(OPENAI_VOICE_ALIASES),
        }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_speech_request(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)

            try:
                request_data = json.loads(post_data.decode('utf-8'))
            except json.JSONDecodeError:
                self.send_error_response(400, "Invalid JSON")
                return

            input_text = request_data.get('input')
            voice = request_data.get('voice', 'af_heart')
            response_format = request_data.get('response_format', 'mp3')
            speed = request_data.get('speed', 1.0)

            if not input_text:
                self.send_error_response(400, "Missing required parameter 'input'")
                return

            if voice in AVAILABLE_VOICES:
                speaker = voice
            elif voice in OPENAI_VOICE_ALIASES:
                speaker = DEFAULT_VOICE
            else:
                logger.warning(
                    f"Unknown voice '{voice}' for lang '{config.LANGUAGE}'; "
                    f"falling back to '{DEFAULT_VOICE}'"
                )
                speaker = DEFAULT_VOICE

            logger.info(f"OpenAI API: Generating speech for text: {input_text[:100]}...")
            samples = generate_speech(text=input_text, speaker=speaker, speed=speed)
            if samples is None:
                self.send_error_response(500, "Speech synthesis failed")
                return

            audio_data, content_type = encode_audio(samples, response_format)

            visemes = compute_visemes(samples)
            visemes_b64 = None
            if visemes is not None:
                try:
                    visemes_b64 = base64.b64encode(
                        json.dumps(visemes, separators=(',', ':')).encode()
                    ).decode('ascii')
                except Exception as e:
                    logger.warning("failed to base64-encode visemes: %s", e)

            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(audio_data)))
            if visemes_b64:
                self.send_header('X-Visemes', visemes_b64)
                # Browser clients (dashboard) can only read X-* response
                # headers when explicitly exposed via CORS.
                self.send_header('Access-Control-Expose-Headers', 'X-Visemes')
            self.end_headers()
            self.wfile.write(audio_data)

            cue_count = len(visemes.get('mouthCues', [])) if visemes else 0
            logger.info(
                f"OpenAI API: served {len(audio_data)} bytes as {content_type} "
                f"(requested response_format='{response_format}', "
                f"visemes={'omitted' if visemes is None else f'{cue_count} cues'})"
            )

        except Exception as e:
            logger.error(f"Error in OpenAI speech endpoint: {e}")
            self.send_error_response(500, f"Internal server error: {str(e)}")

    def send_error_response(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error_response = {"error": {"message": message, "type": "invalid_request_error"}}
        self.wfile.write(json.dumps(error_response).encode())


def generate_speech(text, speaker="af_heart", speed=1.0):
    """Run Kokoro on ``text`` and return concatenated float PCM samples.

    Returns ``None`` on failure. Also writes a WAV copy under ``AUDIO_DIR``
    for offline inspection — not used by the response path.
    """
    logger.info(f"Generating speech for text: {text}")
    unique_id = str(uuid.uuid4().int)[:4]
    filename = f"{int(time.time())}-{unique_id}.wav"
    filepath = os.path.join(config.AUDIO_DIR, filename)

    try:
        generator = pipeline(text, voice=speaker, speed=speed)

        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            logger.error("Kokoro produced no audio chunks")
            return None

        # Kokoro may yield torch tensors; normalize to a float32 numpy array
        # so downstream encode_audio can clip/scale without surprises.
        as_np = [
            a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
            for a in audio_chunks
        ]
        samples = as_np[0] if len(as_np) == 1 else np.concatenate(as_np)
        samples = np.asarray(samples, dtype=np.float32)

        try:
            sf.write(filepath, samples, SAMPLE_RATE)
            logger.info(f"Generated file: {filepath}")
        except Exception as e:
            logger.warning(f"Debug WAV write failed ({filepath}): {e}")

        return samples

    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return None


def start_openai_server():
    openai_port = getattr(config, 'OPENAI_API_PORT', 6005)
    openai_host = getattr(config, 'OPENAI_API_HOST', '0.0.0.0')

    httpd = HTTPServer((openai_host, openai_port), OpenAICompatibleHandler)
    logger.info(f"OpenAI-compatible API server running on {openai_host}:{openai_port}")
    logger.info(f"Endpoint: http://{openai_host}:{openai_port}/v1/audio/speech")
    httpd.serve_forever()


if __name__ == "__main__":
    try:
        start_openai_server()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", exc_info=True)
        raise
