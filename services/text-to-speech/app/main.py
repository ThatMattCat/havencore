import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
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

logger = logger_module.get_logger('custom')

os.makedirs(config.AUDIO_DIR, exist_ok=True)


device = "cuda:0" if torch.cuda.is_available() else "cpu"
mp="./models/xtts_v2"
tts = TTS(model_path=f"{mp}",config_path=f"{mp}/config.json",progress_bar=False).to(device)

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

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", exc_info=True)