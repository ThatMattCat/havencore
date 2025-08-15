import os
from http.server import HTTPServer, SimpleHTTPRequestHandler, BaseHTTPRequestHandler
import threading
import config
import time
import gradio as gr
import torch
import uuid
import json
from kokoro import KPipeline
import soundfile as sf

if not config.SOLO:
    from shared.scripts.trace_id import with_trace, get_trace_id, set_trace_id
    import shared.scripts.logger as logger_module
    logger = logger_module.get_logger('loki')
else:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('text-to-speech')

os.makedirs(config.AUDIO_DIR, exist_ok=True)

pipeline = KPipeline(lang_code=config.LANGUAGE, device=config.MODEL_DEVICE if torch.cuda.is_available() else 'cpu')

class OpenAICompatibleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/audio/speech":
            self.handle_speech_request()
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

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
            model = request_data.get('model', 'tts-1')
            voice = request_data.get('voice', 'af_heart')
            response_format = request_data.get('response_format', 'mp3')
            speed = request_data.get('speed', 1.0)

            if not input_text:
                self.send_error_response(400, "Missing required parameter 'input'")
                return

            voice_mapping = {
                'alloy': 'af_heart',
                'echo': 'af_heart',
                'fable': 'af_heart',
                'onyx': 'af_heart',
                'nova': 'af_heart',
                'shimmer': 'af_heart'
            }

            speaker = voice_mapping.get(voice, 'af_heart')

            logger.info(f"OpenAI API: Generating speech for text: {input_text[:100]}...")
            filepath, filename = generate_speech(
                text=input_text,
                speaker=speaker,
                speed=speed
            )

            with open(filepath, 'rb') as audio_file:
                audio_data = audio_file.read()

            self.send_response(200)
            
            content_type_map = {
                'mp3': 'audio/mpeg',
                'wav': 'audio/wav',
                'opus': 'audio/opus',
                'aac': 'audio/aac',
                'flac': 'audio/flac',
                'pcm': 'audio/pcm'
            }
            
            content_type = content_type_map.get(response_format, 'audio/wav')
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(audio_data)))
            self.end_headers()

            self.wfile.write(audio_data)
            
            logger.info(f"OpenAI API: Successfully served audio for request")

        except Exception as e:
            logger.error(f"Error in OpenAI speech endpoint: {e}")
            self.send_error_response(500, f"Internal server error: {str(e)}")

    def send_error_response(self, status_code, message):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error_response = {
            "error": {
                "message": message,
                "type": "invalid_request_error"
            }
        }
        self.wfile.write(json.dumps(error_response).encode())

class RestrictedHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        return SimpleHTTPRequestHandler.do_GET(self)

def generate_speech(text, speaker="af_heart", speed=1.0) -> tuple[str, str]:
    logger.info(f"Generating speech for text: {text}")
    unique_id = str(uuid.uuid4().int)[:4]
    filename = f"{int(time.time())}-{unique_id}.wav"
    filepath = os.path.join(config.AUDIO_DIR, filename)
    
    try:
        generator = pipeline(text, voice=speaker, speed=speed)
        
        audio_chunks = []
        for i, (_, _, audio) in enumerate(generator):
            audio_chunks.append(audio)
        
        if len(audio_chunks) == 1:
            sf.write(filepath, audio_chunks[0], 24000)
            logger.info(f"Generated file: {filepath}")
        else:
            import numpy as np
            concatenated_audio = np.concatenate(audio_chunks)
            sf.write(filepath, concatenated_audio, 24000)
            logger.info(f"Generated file: {filepath}")
        
        return filepath, filename
            
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return None, None

def start_server():
    os.chdir(config.AUDIO_DIR)
    httpd = HTTPServer((config.SERVER_HOST, config.SERVER_PORT), RestrictedHandler)
    logger.info(f"Serving audio files on port {config.SERVER_PORT}")
    httpd.serve_forever()

def start_openai_server():
    """Start the OpenAI-compatible API server"""

    openai_port = getattr(config, 'OPENAI_API_PORT', 6005)
    openai_host = getattr(config, 'OPENAI_API_HOST', '0.0.0.0')
    
    httpd = HTTPServer((openai_host, openai_port), OpenAICompatibleHandler)
    logger.info(f"OpenAI-compatible API server running on {openai_host}:{openai_port}")
    logger.info(f"Endpoint: http://{openai_host}:{openai_port}/v1/audio/speech")
    httpd.serve_forever()

def generate_audio(text, speaker="af_heart"):
    logger.debug(f"Processing audio request for text: {text}")
    filepath, filename = generate_speech(text=text, speaker=speaker)
    
    audio_url = config.BASE_URL + filename
    
    logger.info(f"Audio generated and shared: {audio_url}")
    return audio_url, filepath

#@with_trace
def main():
    try:
        logger.info("Starting Text-To-Speech Service")
        iface = gr.Interface(
            fn=generate_audio,
            inputs=[
                gr.Textbox(label="Enter your text")
            ],
            outputs=[gr.Textbox(label="Download URL"),
                     gr.Audio(label="Audio Output")],
            title="Text-To-Speech Converter",
            description="Enter the text to convert to audio."
        )
        logger.info("Starting TTS Gradio Interface")#, extra={"trace_id": trace_id})
        iface.launch(server_name="0.0.0.0", server_port=6004)
        logger.info("Exiting Text-To-Speech Service")
    except Exception as e:
        logger.error(f"Error in TTS main(): {e}")
        raise e

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

openai_server_thread = threading.Thread(target=start_openai_server, daemon=True)
openai_server_thread.start()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", exc_info=True)