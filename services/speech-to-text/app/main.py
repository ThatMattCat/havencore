import os
import asyncio
import logging
import tempfile
import uuid
from typing import Optional

from aiohttp import web

import config
import shared.configs.shared_config as shared_config
import shared.scripts.logger as logger_module
from shared.scripts.trace_id import with_trace, get_trace_id, set_trace_id
from whisper_streaming.whisper_online import FasterWhisperASR

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = shared_config.STT_DEVICE
logger = logger_module.get_logger('loki')


class WhisperTranscriber:
    @with_trace
    def __init__(self):
        self.asr = FasterWhisperASR(shared_config.SRC_LAN, config.WHISPER_MODEL)
        self.asr.transcribe(config.WARMUP_FILE)

    @with_trace
    async def transcribe_file(self, audio_file_path: str, language: Optional[str] = None) -> str:
        """Transcribe an audio file using FasterWhisper."""
        trace_id = get_trace_id()
        file_asr = FasterWhisperASR(
            language or shared_config.SRC_LAN,
            config.WHISPER_MODEL,
        )

        logger.info(f"Transcribing file: {audio_file_path}", extra={'trace_id': trace_id})

        segments = await asyncio.to_thread(file_asr.transcribe, audio_file_path)

        transcript_parts = []
        for segment in segments:
            if hasattr(segment, 'text'):
                transcript_parts.append(segment.text)
            elif isinstance(segment, tuple) and len(segment) >= 3:
                transcript_parts.append(segment[2])
            elif isinstance(segment, str):
                transcript_parts.append(segment)
            else:
                logger.warning(f"Unknown segment format: {type(segment)}", extra={'trace_id': trace_id})

        transcript = " ".join(transcript_parts).strip()
        logger.info(f"File transcription complete: {transcript[:100]}...", extra={'trace_id': trace_id})
        return transcript


class TranscriptionAPIHandler:
    """OpenAI-compatible transcription HTTP handler."""

    def __init__(self, transcriber: WhisperTranscriber):
        self.transcriber = transcriber

    async def handle_transcription(self, request: web.Request) -> web.Response:
        """POST /v1/audio/transcriptions (OpenAI-compatible)."""
        trace_id = str(uuid.uuid4())
        set_trace_id(trace_id)

        try:
            reader = await request.multipart()

            audio_file = None
            model = None
            language = None
            response_format = "json"

            async for field in reader:
                field_name = field.name

                if field_name == 'file':
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.audio') as tmp_file:
                        while True:
                            chunk = await field.read_chunk()
                            if not chunk:
                                break
                            tmp_file.write(chunk)
                        audio_file = tmp_file.name

                elif field_name == 'model':
                    model = await field.text()

                elif field_name == 'language':
                    language = await field.text()

                elif field_name == 'response_format':
                    response_format = await field.text()

            if not audio_file:
                return web.json_response(
                    {"error": {"message": "No audio file provided", "type": "invalid_request_error"}},
                    status=400
                )

            logger.info(
                f"Received transcription request - Model: {model}, Language: {language}, Format: {response_format}",
                extra={'trace_id': trace_id},
            )

            transcript = await self.transcriber.transcribe_file(audio_file, language)

            try:
                os.unlink(audio_file)
            except Exception as e:
                logger.warning(f"Could not delete temp file {audio_file}: {e}", extra={'trace_id': trace_id})

            if response_format == "text":
                return web.Response(text=transcript, content_type='text/plain')
            if response_format == "srt":
                srt_content = f"1\n00:00:00,000 --> 00:00:10,000\n{transcript}\n"
                return web.Response(text=srt_content, content_type='text/plain')
            if response_format == "vtt":
                vtt_content = f"WEBVTT\n\n00:00:00.000 --> 00:00:10.000\n{transcript}\n"
                return web.Response(text=vtt_content, content_type='text/vtt')
            if response_format == "verbose_json":
                return web.json_response({
                    "task": "transcribe",
                    "language": language or shared_config.SRC_LAN,
                    "duration": None,
                    "text": transcript,
                    "segments": [],
                })
            return web.json_response({"text": transcript})

        except Exception as e:
            logger.error(f"Error handling transcription request: {e}", exc_info=True, extra={'trace_id': trace_id})
            return web.json_response(
                {"error": {"message": str(e), "type": "internal_error"}},
                status=500
            )

    async def handle_translations(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"error": {"message": "Translation endpoint not yet implemented", "type": "not_implemented"}},
            status=501
        )


async def create_http_app(transcriber: WhisperTranscriber) -> web.Application:
    app = web.Application()
    handler = TranscriptionAPIHandler(transcriber)

    app.router.add_post('/v1/audio/transcriptions', handler.handle_transcription)
    app.router.add_post('/v1/audio/translations', handler.handle_translations)

    async def health_check(request):
        return web.json_response({"status": "healthy"})

    app.router.add_get('/health', health_check)
    return app


async def main():
    transcriber = WhisperTranscriber()

    http_app = await create_http_app(transcriber)
    runner = web.AppRunner(http_app)
    await runner.setup()

    http_port = int(os.environ.get('HTTP_API_PORT', 6001))
    site = web.TCPSite(runner, '0.0.0.0', http_port)
    await site.start()
    logger.info(f"HTTP API server started on port {http_port}")
    logger.info(f"OpenAI API endpoint available at: http://0.0.0.0:{http_port}/v1/audio/transcriptions")

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
