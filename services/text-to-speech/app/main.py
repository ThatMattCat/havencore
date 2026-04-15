import os
import time
import uuid
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

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
            filepath, _ = generate_speech(text=input_text, speaker=speaker, speed=speed)

            with open(filepath, 'rb') as audio_file:
                audio_data = audio_file.read()

            content_type_map = {
                'mp3': 'audio/mpeg',
                'wav': 'audio/wav',
                'opus': 'audio/opus',
                'aac': 'audio/aac',
                'flac': 'audio/flac',
                'pcm': 'audio/pcm',
            }
            content_type = content_type_map.get(response_format, 'audio/wav')

            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(audio_data)))
            self.end_headers()
            self.wfile.write(audio_data)

            logger.info("OpenAI API: Successfully served audio for request")

        except Exception as e:
            logger.error(f"Error in OpenAI speech endpoint: {e}")
            self.send_error_response(500, f"Internal server error: {str(e)}")

    def send_error_response(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error_response = {"error": {"message": message, "type": "invalid_request_error"}}
        self.wfile.write(json.dumps(error_response).encode())


def generate_speech(text, speaker="af_heart", speed=1.0) -> tuple[str, str]:
    logger.info(f"Generating speech for text: {text}")
    unique_id = str(uuid.uuid4().int)[:4]
    filename = f"{int(time.time())}-{unique_id}.wav"
    filepath = os.path.join(config.AUDIO_DIR, filename)

    try:
        generator = pipeline(text, voice=speaker, speed=speed)

        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        if len(audio_chunks) == 1:
            sf.write(filepath, audio_chunks[0], 24000)
        else:
            import numpy as np
            concatenated_audio = np.concatenate(audio_chunks)
            sf.write(filepath, concatenated_audio, 24000)

        logger.info(f"Generated file: {filepath}")
        return filepath, filename

    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return None, None


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
