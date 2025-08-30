import gradio as gr
import requests
import json
import base64
from typing import List, Dict, Any, Optional
import os
import shutil
from pathlib import Path

# Server configuration - using Docker service name and internal port
SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://iav-to-text-vllm:8100/v1/chat/completions")

UPLOAD_DIR = Path("/app/uploads")  
UPLOAD_DIR.mkdir(exist_ok=True)

def save_uploaded_file(file_path: Optional[str], file_type: str) -> Optional[str]:
    """Save uploaded file to our controlled directory."""
    if not file_path or not os.path.exists(file_path):
        return None
    
    # Get file extension
    ext = Path(file_path).suffix
    
    # Create unique filename
    import time
    timestamp = int(time.time() * 1000)
    new_filename = f"{file_type}_{timestamp}{ext}"
    new_path = UPLOAD_DIR / new_filename
    
    # Copy file to our directory
    shutil.copy2(file_path, new_path)
    
    return str(new_path.absolute())

def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 string."""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')

def create_message_content(
    text: Optional[str] = None,
    image: Optional[str] = None,
    audio: Optional[str] = None,
    video: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Create message content array based on provided inputs."""
    content = []
    
    # Add text if provided
    if text and text.strip():
        content.append({"type": "text", "text": text})
    
    # Add image if provided
    if image and os.path.exists(image):
        # Use file:// URL format for local files
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"file://{os.path.abspath(image)}"}
        })
    
    # Add audio if provided
    if audio and os.path.exists(audio):
        # Use file:// URL format for local files
        content.append({
            "type": "audio_url",
            "audio_url": {"url": f"file://{os.path.abspath(audio)}"}
        })
    
    # Add video if provided
    if video and os.path.exists(video):
        # Use file:// URL format for local files
        content.append({
            "type": "video_url",
            "video_url": {"url": f"file://{os.path.abspath(video)}"}
        })
    
    return content

def send_to_vllm(
    text_input: str,
    image_input: Optional[str],
    audio_input: Optional[str],
    video_input: Optional[str],
    system_prompt: str,
    temperature: float,
    max_tokens: int
) -> str:
    """Send request to vLLM server and get response."""
    
    try:
        # Save uploaded files to our controlled directory
        saved_image = save_uploaded_file(image_input, "image")
        saved_audio = save_uploaded_file(audio_input, "audio")
        saved_video = save_uploaded_file(video_input, "video")
        
        # Create user message content
        user_content = create_message_content(
            text=text_input,
            image=saved_image,
            audio=saved_audio,
            video=saved_video
        )
        
        if not user_content:
            return "Please provide at least one input (text, image, audio, or video)."
        
        # Construct messages
        messages = []
        
        # Add system message if provided
        if system_prompt and system_prompt.strip():
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Prepare request payload
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Send request to vLLM server
        response = requests.post(
            SERVER_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract the generated text
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "No response generated."
            
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to vLLM server at {SERVER_URL}. Please check if the server is running."
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The server might be processing a heavy request."
    except requests.exceptions.RequestException as e:
        return f"Error communicating with server: {str(e)}"
    except json.JSONDecodeError as e:
        return f"Error parsing server response: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def clear_inputs():
    """Clear all input fields."""
    return "", None, None, None, ""

# Create Gradio interface
with gr.Blocks(title="Qwen2.5-Omni Multimodal Interface") as app:
    gr.Markdown(
        """
        # 🤖 Qwen2.5-Omni Multimodal Interface
        
        This interface connects to a vLLM server running Qwen2.5-Omni-7B model.
        The model can process text, images, audio, and video inputs.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input components
            system_prompt = gr.Textbox(
                label="System Prompt",
                placeholder="You are a helpful assistant...",
                value="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                lines=2
            )
            
            text_input = gr.Textbox(
                label="Text Input",
                placeholder="Enter your text query here...",
                lines=3
            )
            
            image_input = gr.Image(
                label="Image Input",
                type="filepath",
                sources=["upload", "clipboard"]
            )
            
            audio_input = gr.Audio(
                label="Audio Input",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            video_input = gr.Video(
                label="Video Input",
                sources=["upload"]
            )
            
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=4096,
                    value=1024,
                    step=50,
                    label="Max Tokens"
                )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        with gr.Column(scale=1):
            # Output component
            output = gr.Textbox(
                label="Model Response",
                lines=20,
                placeholder="Model response will appear here..."
            )
    
    # Example inputs - simplified without file inputs
    gr.Examples(
        examples=[
            ["What can you see and hear in these media files?"],
            ["Describe this image in detail."],
            ["Transcribe this audio."],
        ],
        inputs=[text_input],
        outputs=output,
        fn=lambda x: "Please upload media files and click Submit to process them.",
        cache_examples=False
    )
    
    # Server status
    gr.Markdown(
        f"""
        ### Server Information
        - **Server URL**: `{SERVER_URL}`
        - **Model**: Qwen2.5-Omni-7B
        - **Supported Inputs**: Text, Image, Audio, Video
        - **Upload Directory**: `{UPLOAD_DIR.absolute()}`
        
        ⚠️ **Important**: Make sure vLLM has read access to: `{UPLOAD_DIR.absolute()}`
        """
    )
    
    # Event handlers
    submit_btn.click(
        fn=send_to_vllm,
        inputs=[
            text_input,
            image_input,
            audio_input,
            video_input,
            system_prompt,
            temperature,
            max_tokens
        ],
        outputs=output
    )
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[text_input, image_input, audio_input, video_input, output]
    )

# Launch the app
if __name__ == "__main__":
    print(f"Starting Gradio app...")
    print(f"vLLM Server URL: {SERVER_URL}")
    print(f"Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"Starting on http://0.0.0.0:8110")
    
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=8110,       # Gradio default port
        share=False,            # Set to True if you want a public URL
        debug=False,            # Disable debug mode for production
        root_path=""#os.getenv("GRADIO_ROOT_PATH", "")  # Support for reverse proxy
    )
    
    print("Gradio app is running!")