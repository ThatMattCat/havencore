import os
from http.server import HTTPServer, SimpleHTTPRequestHandler, BaseHTTPRequestHandler
import threading
import config
from TTS.api import TTS
import tempfile
import logging
from logging.config import dictConfig
from datetime import datetime
import time
import gradio as gr
import torch
import uuid
from shared.scripts.trace_id import with_trace, get_trace_id, set_trace_id
import shared.scripts.logger as logger_module
import shared.configs.shared_config as shared_config
import json
import urllib.parse
from io import BytesIO

logger = logger_module.get_logger('loki')

os.makedirs(config.AUDIO_DIR, exist_ok=True)

device = f"cuda:{shared_config.TTS_DEVICE}" if torch.cuda.is_available() else "cpu"
mp="./models/xtts_v2"
tts = TTS(model_path=f"{mp}",config_path=f"{mp}/config.json",progress_bar=False).to(device)

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

            # Extract required parameters
            input_text = request_data.get('input')
            model = request_data.get('model', 'tts-1')  # OpenAI models: tts-1, tts-1-hd
            voice = request_data.get('voice', 'alloy')  # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
            response_format = request_data.get('response_format', 'mp3')  # mp3, opus, aac, flac, wav, pcm
            speed = request_data.get('speed', 1.0)  # 0.25 to 4.0

            if not input_text:
                self.send_error_response(400, "Missing required parameter 'input'")
                return

            # Map OpenAI voices to your available speakers (customize as needed)
            voice_mapping = {
                'alloy': 'Camilla Holmström',
                'echo': 'Camilla Holmström',
                'fable': 'Camilla Holmström', 
                'onyx': 'Camilla Holmström',
                'nova': 'Camilla Holmström',
                'shimmer': 'Camilla Holmström'
            }
            
            speaker = voice_mapping.get(voice, 'Camilla Holmström')
            
            # Generate audio using existing function
            logger.info(f"OpenAI API: Generating speech for text: {input_text[:100]}...")
            filepath, filename = generate_speech(
                text=input_text,
                speaker=speaker,
                split_sentences=len(input_text) > 200  # Auto-split for longer texts
            )

            # Read the generated audio file
            with open(filepath, 'rb') as audio_file:
                audio_data = audio_file.read()

            # Set appropriate headers
            self.send_response(200)
            
            # Set content type based on response format
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

            # Send audio data
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

    def log_message(self, format, *args):
        # Override to use your logger instead of default logging
        logger.info(f"OpenAI API Request: {format % args}")

class RestrictedHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        return SimpleHTTPRequestHandler.do_GET(self)

def generate_speech(text, speaker="Camilla Holmström", split_sentences=False):
    logger.info(f"Generating speech for text: {text}")
    unique_id = str(uuid.uuid4().int)[:4]
    filename = f"{int(time.time())}-{unique_id}.wav"
    filepath = os.path.join(config.AUDIO_DIR, filename)

    tts.tts_to_file(text=text,
                    file_path=filepath,
                    speaker=speaker,
                    language="en",
                    split_sentences=split_sentences)
    logger.info(f"Generated file: {filepath}")
    return filepath, filename

def start_server():
    os.chdir(config.AUDIO_DIR)
    httpd = HTTPServer((config.SERVER_HOST, config.SERVER_PORT), RestrictedHandler)
    logger.info(f"Serving audio files on port {config.SERVER_PORT}")
    httpd.serve_forever()

def start_openai_server():
    """Start the OpenAI-compatible API server"""
    # Use a different port for the OpenAI API (you can configure this)
    openai_port = getattr(config, 'OPENAI_API_PORT', 6005)
    openai_host = getattr(config, 'OPENAI_API_HOST', '0.0.0.0')
    
    httpd = HTTPServer((openai_host, openai_port), OpenAICompatibleHandler)
    logger.info(f"OpenAI-compatible API server running on {openai_host}:{openai_port}")
    logger.info(f"Endpoint: http://{openai_host}:{openai_port}/v1/audio/speech")
    httpd.serve_forever()

def generate_audio(text, speaker="Camilla Holmström", split_sentences=False):
    logger.debug(f"Processing audio request for text: {text}")
    filepath, filename = generate_speech(text=text, speaker=speaker, split_sentences=split_sentences)
    
    audio_url = config.BASE_URL + filename
    
    logger.info(f"Audio generated and shared: {audio_url}")
    return audio_url, filepath

#@with_trace
def main():
    # TODO: Use admin api key to load proper model/etc based on model_type
#    trace_id = get_trace_id()
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

# Start both servers
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

openai_server_thread = threading.Thread(target=start_openai_server, daemon=True)
openai_server_thread.start()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", exc_info=True)