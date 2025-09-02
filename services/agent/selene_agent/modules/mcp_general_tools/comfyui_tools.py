"""
Simplified ComfyUI Client - Cleaner async implementation
"""

import json
import asyncio
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import aiohttp
import aiofiles
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import os


@dataclass
class ComfyUIConfig:
    """Configuration for ComfyUI client"""
    server: str = "text-to-image"
    port: int = 8188
    workflow_dir: Path = field(default_factory=lambda: Path("/app/selene_agent/modules/mcp_general_tools/comfyui_workflows"))
    output_dir: Path = field(default_factory=lambda: Path("/app/selene_agent/outputs"))
    timeout: int = 120
    
    def __post_init__(self):
        self.workflow_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.base_url = f"http://{self.server}:{self.port}"


class ComfyUIClient:
    """Simplified ComfyUI client with better async handling"""
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.config = config or ComfyUIConfig()
        self.client_id = str(uuid.uuid4())
        self._session: Optional[aiohttp.ClientSession] = None
    
    @asynccontextmanager
    async def session(self):
        """Context manager for aiohttp session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        try:
            yield self._session
        finally:
            # Don't close here, let __aexit__ handle it
            pass
    
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()
            self._session = None
    
    # Core API Methods
    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow for execution, returns prompt_id"""
        async with self.session() as session:
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }
            async with session.post(f"{self.config.base_url}/prompt", json=payload) as resp:
                result = await resp.json()
                return result.get('prompt_id')
    
    async def get_status(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Check if a prompt has completed"""
        async with self.session() as session:
            async with session.get(f"{self.config.base_url}/history/{prompt_id}") as resp:
                history = await resp.json()
                return history.get(prompt_id)
    
    async def wait_for_completion(self, prompt_id: str) -> Dict[str, Any]:
        """Wait for prompt completion with exponential backoff"""
        start = time.time()
        backoff = 0.5
        
        while time.time() - start < self.config.timeout:
            result = await self.get_status(prompt_id)
            if result:
                return result
            
            await asyncio.sleep(min(backoff, 5))
            backoff *= 1.5
        
        raise TimeoutError(f"Prompt {prompt_id} timed out after {self.config.timeout}s")
    
    async def download_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download a generated image"""
        async with self.session() as session:
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type
            }
            async with session.get(f"{self.config.base_url}/view", params=params) as resp:
                return await resp.read()
    
    # Workflow Management
    async def save_workflow(self, workflow: Dict[str, Any], name: str):
        """Save workflow to disk"""
        filepath = self.config.workflow_dir / f"{name}.json"
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(workflow, indent=2))
    
    async def load_workflow(self, name: str) -> Dict[str, Any]:
        """Load workflow from disk"""
        filepath = self.config.workflow_dir / f"{name}.json"
        try:
            async with aiofiles.open(filepath, 'r') as f:
                return json.loads(await f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"Workflow file '{filepath}' not found.")
        except Exception as e:
            raise RuntimeError(f"Failed to load workflow '{filepath}': {e}")
    
    # High-level Methods
    async def generate(self, 
                      workflow: Dict[str, Any],
                      wait: bool = True) -> Dict[str, Any]:
        """Generate images from workflow"""
        prompt_id = await self.queue_prompt(workflow)
        
        if not wait:
            return {"prompt_id": prompt_id, "status": "queued"}
        
        result = await self.wait_for_completion(prompt_id)
        
        # Extract and download images
        images = self._extract_image_info(result)
        downloaded = []
        
        for img in images:
            data = await self.download_image(
                img['filename'], 
                img.get('subfolder', ''),
                img.get('type', 'output')
            )
            
            # Save locally
            output_path = self.config.output_dir / img['filename']
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(data)
            
            downloaded.append({
                "path": str(output_path),
                "url": f"http://{os.getenv('HOST_IP_ADDRESS')}:6006/outputs/{img['filename']}",
                "filename": img['filename']
            })
        
        return {
            "prompt_id": prompt_id,
            "images": downloaded,
            "history": result
        }
    
    @staticmethod
    def _extract_image_info(history: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract image information from history"""
        images = []
        outputs = history.get('outputs', {})
        
        for node_output in outputs.values():
            if 'images' in node_output:
                images.extend(node_output['images'])
        
        return images


class WorkflowBuilder:
    """Simplified workflow manipulation"""
    
    @staticmethod
    def update_prompts(workflow: Dict[str, Any],
                      positive: str,
                      negative: Optional[str] = None,
                      seed: Optional[int] = None) -> Dict[str, Any]:
        """Update common workflow parameters"""
        # Work on a copy
        workflow = json.loads(json.dumps(workflow))
        
        # Find text encode nodes and categorize them
        text_nodes = []
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                text_nodes.append((node_id, node))
        
        # Sort by node ID (usually positive comes first)
        text_nodes.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
        
        # Update prompts
        if text_nodes and positive:
            text_nodes[0][1]["inputs"]["text"] = positive
        
        if len(text_nodes) > 1 and negative:
            text_nodes[1][1]["inputs"]["text"] = negative
        
        # Update seed if provided
        if seed is not None:
            for node in workflow.values():
                if node.get("class_type") == "KSampler" and "inputs" in node:
                    node["inputs"]["seed"] = seed
        
        return workflow
    
    @staticmethod
    def update_node(workflow: Dict[str, Any], 
                   node_id: str, 
                   updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update specific node inputs"""
        workflow = json.loads(json.dumps(workflow))
        
        if node_id in workflow and "inputs" in workflow[node_id]:
            workflow[node_id]["inputs"].update(updates)
        
        return workflow
    
    @staticmethod
    def batch_update(workflow: Dict[str, Any],
                    updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Update multiple nodes at once"""
        workflow = json.loads(json.dumps(workflow))
        
        for node_id, node_updates in updates.items():
            if node_id in workflow and "inputs" in workflow[node_id]:
                workflow[node_id]["inputs"].update(node_updates)
        
        return workflow


class SimpleComfyUI:
    """High-level interface for common operations"""
    
    def __init__(self, server: str = "text-to-image:8188"):
        host, port = server.split(':') if ':' in server else (server, "8188")
        self.config = ComfyUIConfig(server=host, port=int(port))
        self.client = ComfyUIClient(self.config)
        self.builder = WorkflowBuilder()
        
        # Default negative prompt for quality
        self.default_negative = (
            "lowres, bad anatomy, bad hands, text, error, missing fingers, "
            "extra digit, cropped, worst quality, low quality, jpeg artifacts, "
            "signature, watermark, blurry"
        )
    
    async def __aenter__(self):
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, *args):
        await self.client.__aexit__(*args)
    
    async def text_to_image(self,
                          prompt: str,
                          workflow_name: str = "default",
                          negative: Optional[str] = None,
                          seed: Optional[int] = None,
                          **kwargs) -> Dict[str, Any]:
        """Simple text-to-image generation"""
        # Load workflow
        workflow = await self.client.load_workflow(workflow_name)
        
        # Update with prompts
        workflow = self.builder.update_prompts(
            workflow,
            positive=prompt,
            negative=negative or self.default_negative,
            seed=seed
        )
        
        # Apply any additional node updates
        if kwargs:
            workflow = self.builder.batch_update(workflow, kwargs)
        
        # Generate
        return await self.client.generate(workflow)
    
    async def import_workflow(self, 
                            source_path: str, 
                            save_as: str,
                            analyze: bool = True) -> Dict[str, Any]:
        """Import a workflow from ComfyUI export"""
        async with aiofiles.open(source_path, 'r') as f:
            workflow = json.loads(await f.read())
        
        await self.client.save_workflow(workflow, save_as)
        
        if analyze:
            return self.analyze_workflow(workflow)
        return workflow
    
    @staticmethod
    def analyze_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow structure"""
        analysis = {
            "nodes": {},
            "connections": [],
            "prompts": [],
            "samplers": []
        }
        
        for node_id, node in workflow.items():
            class_type = node.get("class_type", "Unknown")
            analysis["nodes"][node_id] = class_type
            
            if class_type == "CLIPTextEncode":
                text = node.get("inputs", {}).get("text", "")
                analysis["prompts"].append({
                    "id": node_id,
                    "preview": text[:50] + "..." if len(text) > 50 else text
                })
            elif class_type == "KSampler":
                analysis["samplers"].append(node_id)
        
        return analysis


# Example usage
async def example_simple():
    """Simple usage example"""
    async with SimpleComfyUI("localhost:8188") as comfy:
        # Generate an image
        result = await comfy.text_to_image(
            prompt="a beautiful sunset over mountains, digital art",
            workflow_name="default",
            seed=42
        )
        
        print(f"Generated {len(result['images'])} images")
        for img in result['images']:
            print(f"  - {img['path']}")


async def example_advanced():
    """Advanced usage example"""
    config = ComfyUIConfig(
        server="localhost",
        port=8188,
        timeout=180
    )
    
    async with ComfyUIClient(config) as client:
        # Load and customize workflow
        workflow = await client.load_workflow("my_workflow")
        
        # Direct node manipulation
        workflow = WorkflowBuilder.update_node(
            workflow,
            node_id="3",
            updates={"steps": 30, "cfg": 8.0}
        )
        
        # Generate
        result = await client.generate(workflow)
        print(f"Completed: {result['prompt_id']}")


if __name__ == "__main__":
    # Run simple example
    asyncio.run(example_simple())