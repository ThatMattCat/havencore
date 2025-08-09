import os
from enum import Enum
import asyncio
from functools import partial
import websockets
import json
import logging
import config
import numpy as np
from whisper_streaming.whisper_online import FasterWhisperASR, OnlineASRProcessor
import time
from collections import deque
from asyncio import Queue
import requests
import aiohttp
from aiohttp import ClientSession
from gradio_client import Client
from shared.scripts.trace_id import with_trace, get_trace_id, set_trace_id
import shared.scripts.logger as logger_module
import shared.configs.shared_config as shared_config

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = shared_config.STT_DEVICE
logger = logger_module.get_logger('loki')

class WSMessages(Enum):
    AUDIO_TYPE = "AUDIO"
    CONTROL_TYPE = "CONTROL"
    START_MSG = "start"
    STOP_MSG = "stop"

class OrderedAudioProcessor:
    @with_trace
    def __init__(self):
        os.makedirs(config.TRANSCRIPT_FOLDER, exist_ok=True)
        os.makedirs(config.AUDIO_FOLDER, exist_ok=True)
        self.asr = FasterWhisperASR(shared_config.SRC_LAN, config.WHISPER_MODEL)
        self.asr.transcribe(config.WARMUP_FILE)
        #asr.use_vad()
        self.online = OnlineASRProcessor(self.asr)
        self.source_ip = shared_config.SOURCE_IP
        # TODO: Automate sizes based on client-provided rates/etc
        self.chunk_size = 16000  # 0.5 seconds of 16kHz audio with 2 bytes per sample
        self.max_buffer_size = 10 * 1024 * 1024  # 10MB
        self.audio_buffer = bytearray()
        self.processing_queue = Queue()
        self.processor_task = None
        self.transcriptions = []
        self.is_processing = False

    @with_trace
    async def handle_audio_data(self, data):
        trace_id = get_trace_id()
        try:
            await self.add_to_buffer(data)
            await self.enqueue_chunks()
        except Exception as e:
            logging.error(f"Error handling audio data: {e}", exc_info=True, extra={'trace_id': trace_id})

    @with_trace
    async def add_to_buffer(self, data):
        trace_id = get_trace_id()
        self.audio_buffer.extend(data)
        if len(self.audio_buffer) > self.max_buffer_size:
            trim_size = int(self.max_buffer_size * 0.98)
            self.audio_buffer = self.audio_buffer[-trim_size:]
            logger.warning(f"Buffer too large, trimmed. New size: {len(self.audio_buffer)} out of {self.max_buffer_size} max", extra={'trace_id': trace_id})

    @with_trace
    async def enqueue_chunks(self):
        trace_id = get_trace_id()
        while len(self.audio_buffer) >= self.chunk_size:
            chunk = self.audio_buffer[:self.chunk_size]
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            await self.processing_queue.put(chunk)
            await asyncio.sleep(0.01)

    @with_trace
    async def process_chunks(self):
        try:
            trace_id = get_trace_id()
            while True:
                chunk = await self.processing_queue.get()
                if chunk is None:
                    logger.info("Received signal ('None' queue item) to stop processing", extra={'trace_id': trace_id})
                    break
                start, end, text = await asyncio.to_thread(self._process_chunk, chunk) # result= 'start' 'end' 'text' eg: 0.0 0.5 'hello'
                if text != 'AABBCCDD':
                    self.transcriptions.append((start,end,text))
                self.processing_queue.task_done()
                await asyncio.sleep(0.01)
            self.is_processing = False
            logger.info("Processing complete", extra={'trace_id': trace_id})
            await self.on_transcription_complete()

        except Exception as e:
            logging.error(f"Error in process_chunks: {e}", exc_info=True, extra={'trace_id': trace_id})

    @with_trace
    def _process_chunk(self, chunk):
        trace_id = get_trace_id()
        try:
            int_audio = np.frombuffer(chunk, dtype=np.int16)
            float_audio = int_audio.astype(np.float32) / 32768.0
            logger.debug(f"Attempting transcription of chunk. Chunk Size: {len(chunk)}", extra={'trace_id': get_trace_id()})
            self.online.insert_audio_chunk(float_audio)
            transcription = self.online.process_iter()
            if transcription and transcription != (None, None, '') and transcription != '':
                start, end, text = transcription
                logging.debug(f"New Chunk Transcription: {start} {end} {text}", extra={'trace_id': trace_id})
                return (start, end, text)
            else:
                return (None, None, 'AABBCCDD')
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}", exc_info=True, extra={'trace_id': trace_id})
            

    @with_trace
    async def end_transcription(self):
        trace_id = get_trace_id()
        await self.processing_queue.put(None)  # Signal end of processing
        logger.info("Stream stopped. Waiting for processing to complete...", extra={'trace_id': trace_id})

    @with_trace
    async def send_transcription(self, transcript, source_ip):
        prefix = "This message was transcribed from audio so it may not be perfectly accurate: "
        transcript = prefix + transcript
        trace_id = get_trace_id()
        
        text_client = Client(f"http://{shared_config.IP_ADDRESS}:6002/") #the AI Agent
        tts_client = Client(f"http://{shared_config.IP_ADDRESS}:6004/")

        try:
            logger.debug(f"Sending transcription to text inference: {transcript}", extra={'trace_id': trace_id})
            loop = asyncio.get_event_loop()
            response_data = await loop.run_in_executor(
                None,
                lambda: text_client.predict(
                    transcript,
                    api_name="/predict"
                )
            )
            logger.debug(f"Agent Response received")
            
            logger.debug(f"Converting text response to audio: {response_data}", extra={'trace_id': trace_id})
            audio_url = await loop.run_in_executor(
                None,
                lambda: tts_client.predict(
                    response_data,
                    api_name="/predict"
                )
            )
            logger.debug(f"Text-to-Speech Response received")
            url = audio_url[0]
            logger.debug(f"Sending audio URL to speaker via WebSocket. URL: {url}", extra={'trace_id': trace_id})
            wsmessage = json.dumps({"url": url, "trace_id": trace_id})
            await self.ws.send(wsmessage)
            
        except Exception as e:
            logger.error(f"Error in send_transcription: {str(e)}", extra={'trace_id': trace_id})
            raise

    @with_trace
    async def begin_transcription(self):
        trace_id = get_trace_id()
        self.processor_task = asyncio.create_task(self.process_chunks())
        self.is_processing = True

    @with_trace
    async def handle_control(self, data):
        try:
            message = data['message']
            trace_id = data['trace_id']
            source_ip = data['source_ip']
            if message == WSMessages.START_MSG.value:
                logger.info("Received start signal.", extra={'trace_id': trace_id})
                set_trace_id(trace_id)
                self.source_ip = source_ip
                await self.begin_transcription()
            elif message == WSMessages.STOP_MSG.value:
                self.source_ip = source_ip
                logger.info("Received finish signal.", extra={'trace_id': trace_id})
                await self.end_transcription()
                self.is_processing = False
            else:
                logger.warning(f"Unknown control message: {message}", extra={'trace_id': trace_id})
        except KeyError as e:
            logger.error(f"KeyError in control message: {e}", extra={'trace_id': get_trace_id()})

    @with_trace
    async def handle_stream(self, websocket, path=None):
        self.ws = websocket #TODO: Not great..websocket handled by outside code
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                        await self.handle_audio_data(message)
                elif isinstance(message, str):
                    data = json.loads(message)
                    if 'type' in data:
                        type = data['type']
                    else:
                        logger.warning("Message type not found in JSON message data")
                        continue
                    if type == WSMessages.CONTROL_TYPE.value:
                        await self.handle_control(data)
                    else:
                        logger.warning(f"Unknown message 'type' field: {type}")
                        continue
                else:
                    logger.warning(f"Unknown message content type: {type(message)}")
                    await self.handle_audio_data(message) #probably audio anyways
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            if self.is_processing:
                await self.end_transcription()
                self.is_processing = False

    @with_trace
    async def on_transcription_complete(self):
        trace_id = get_trace_id()
        transcription = "".join(t[2] for t in self.transcriptions)
        logger.info(f"Final transcription: {transcription}", extra={'trace_id': trace_id})
        test = self.online.finish()
        await self.send_transcription(transcription, self.source_ip)
        await self.reset()

    @with_trace
    async def reset(self):
        self.audio_buffer.clear()
        self.processing_queue = Queue()
        self.transcriptions.clear()
        self.online.init()
        self.processor_task = None

async def main():
    processor = OrderedAudioProcessor()
    server = await websockets.serve(processor.handle_stream, "0.0.0.0", 6000)

    logger.info("Server started. Waiting for connection...")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "This event loop is already running" in str(e):
            asyncio.create_task(main())
        else:
            raise