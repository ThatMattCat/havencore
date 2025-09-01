"""
ComfyUI Workflow Template System
Load saved workflows and use them as templates with dynamic prompts
"""

import json
import urllib.request
import urllib.parse
import uuid
import time
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

class ComfyUIWorkflowClient:
    def __init__(self, server_address: str = "127.0.0.1:8188", workflow_dir: str = "./workflows"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.workflow_dir = Path(workflow_dir)
        self.workflow_dir.mkdir(exist_ok=True)
        
    def queue_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a prompt for execution"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())
    
    def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Retrieve generated image"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()
    
    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt"""
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())
    
    def wait_for_completion(self, prompt_id: str, timeout: int = 120) -> Dict[str, Any]:
        """Wait for a prompt to complete and return the results"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(0.5)
        
        raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout} seconds")
    
    def save_workflow(self, workflow: Dict[str, Any], name: str):
        """Save a workflow to disk for reuse"""
        filepath = self.workflow_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(workflow, f, indent=2)
        print(f"Workflow saved to {filepath}")
    
    def load_workflow(self, name: str) -> Dict[str, Any]:
        """Load a saved workflow from disk"""
        filepath = self.workflow_dir / f"{name}.json"
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def find_node_by_class(self, workflow: Dict[str, Any], class_type: str) -> List[str]:
        """Find all node IDs of a specific class type in the workflow"""
        nodes = []
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == class_type:
                nodes.append(node_id)
        return nodes
    
    def update_workflow_prompts(self, 
                               workflow: Dict[str, Any], 
                               positive_prompt: str,
                               negative_prompt: Optional[str] = None,
                               seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Update a workflow with new prompts and optional parameters
        This works with most standard ComfyUI workflows
        """
        # Deep copy to avoid modifying original
        import copy
        workflow = copy.deepcopy(workflow)
        
        # Find and update positive prompt nodes (CLIPTextEncode)
        positive_nodes = []
        negative_nodes = []
        
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == "CLIPTextEncode":
                # Check if this is likely a positive or negative prompt
                # Usually negative prompts have "negative" in their connections or lower node numbers
                if "inputs" in node_data and "text" in node_data["inputs"]:
                    current_text = node_data["inputs"]["text"].lower()
                    
                    # Simple heuristic: if current text has negative words, it's probably negative
                    negative_keywords = ["bad", "ugly", "worst", "low quality", "blurry", "negative"]
                    if any(keyword in current_text for keyword in negative_keywords):
                        negative_nodes.append(node_id)
                    else:
                        positive_nodes.append(node_id)
        
        # Update positive prompt (usually the first CLIPTextEncode or the one without negative terms)
        if positive_nodes:
            workflow[positive_nodes[0]]["inputs"]["text"] = positive_prompt
        
        # Update negative prompt if provided
        if negative_prompt is not None and negative_nodes:
            workflow[negative_nodes[0]]["inputs"]["text"] = negative_prompt
        
        # Update seed if provided
        if seed is not None:
            sampler_nodes = self.find_node_by_class(workflow, "KSampler")
            for node_id in sampler_nodes:
                if "seed" in workflow[node_id]["inputs"]:
                    workflow[node_id]["inputs"]["seed"] = seed
        
        return workflow
    
    def update_workflow_advanced(self, 
                                workflow: Dict[str, Any],
                                updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced workflow updates with node ID mapping
        
        Example updates dict:
        {
            "6": {"text": "new positive prompt"},  # Node 6 inputs
            "7": {"text": "new negative prompt"},  # Node 7 inputs
            "3": {"seed": 12345, "steps": 30}      # KSampler settings
        }
        """
        import copy
        workflow = copy.deepcopy(workflow)
        
        for node_id, input_updates in updates.items():
            if node_id in workflow and "inputs" in workflow[node_id]:
                workflow[node_id]["inputs"].update(input_updates)
        
        return workflow
    
    def generate_from_workflow(self,
                              workflow_name: str,
                              positive_prompt: str,
                              negative_prompt: Optional[str] = None,
                              seed: Optional[int] = None,
                              wait: bool = True) -> Dict[str, Any]:
        """
        High-level function to generate an image from a saved workflow
        """
        # Load the workflow
        workflow = self.load_workflow(workflow_name)
        
        # Update with new prompts
        workflow = self.update_workflow_prompts(
            workflow, 
            positive_prompt,
            negative_prompt,
            seed
        )
        
        # Queue the generation
        result = self.queue_prompt(workflow)
        prompt_id = result['prompt_id']
        
        if wait:
            # Wait for completion and return results
            return self.wait_for_completion(prompt_id)
        else:
            return {"prompt_id": prompt_id}
    
    def extract_images_from_history(self, history: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract image information from completion history"""
        images = []
        
        if 'outputs' in history:
            for node_id, node_output in history['outputs'].items():
                if 'images' in node_output:
                    for image in node_output['images']:
                        images.append({
                            'filename': image['filename'],
                            'subfolder': image['subfolder'],
                            'type': image['type'],
                            'node_id': node_id
                        })
        
        return images
    
    def save_image_from_history(self, history: Dict[str, Any], output_path: str) -> List[str]:
        """Save all images from a generation history"""
        saved_paths = []
        images = self.extract_images_from_history(history)
        
        for i, image_info in enumerate(images):
            image_data = self.get_image(
                image_info['filename'],
                image_info['subfolder'],
                image_info['type']
            )
            
            # Create output filename
            if len(images) > 1:
                base, ext = os.path.splitext(output_path)
                filepath = f"{base}_{i}{ext}"
            else:
                filepath = output_path
            
            with open(filepath, 'wb') as f:
                f.write(image_data)
            
            saved_paths.append(filepath)
            print(f"Saved image to {filepath}")
        
        return saved_paths


# Example workflow templates with sensible defaults
class WorkflowTemplates:
    """Pre-configured workflow templates with default values"""
    
    @staticmethod
    def create_from_ui_export(api_format_json_path: str) -> Dict[str, Any]:
        """
        Load a workflow exported from ComfyUI in API format
        
        In ComfyUI:
        1. Create your workflow
        2. Click "Save (API Format)" in the menu
        3. Save the JSON file
        4. Use this method to load it
        """
        with open(api_format_json_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def identify_prompt_nodes(workflow: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze a workflow to identify which nodes handle prompts
        Returns a mapping of node purposes
        """
        prompt_nodes = {}
        
        for node_id, node_data in workflow.items():
            class_type = node_data.get("class_type", "")
            
            if class_type == "CLIPTextEncode":
                # Try to identify if it's positive or negative
                text = node_data.get("inputs", {}).get("text", "").lower()
                
                if any(word in text for word in ["negative", "bad", "ugly", "worst"]):
                    prompt_nodes[node_id] = "negative_prompt"
                else:
                    prompt_nodes[node_id] = "positive_prompt"
            
            elif class_type == "KSampler":
                prompt_nodes[node_id] = "sampler"
            
            elif class_type == "CheckpointLoaderSimple":
                prompt_nodes[node_id] = "model_loader"
        
        return prompt_nodes

class SimpleComfyAgent:
    """Simplified interface for AI agents to use ComfyUI"""
    
    def __init__(self, server: str = "text-to-image:8188", workflow_dir: str = "./comfyui_workflows"):
        self.client = ComfyUIWorkflowClient(server, workflow_dir)
        self.default_negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

    def setup_workflow_from_ui(self, ui_export_path: str, save_as: str):
        """
        One-time setup: Import a workflow from ComfyUI
        
        Steps:
        1. Design your workflow in ComfyUI web interface
        2. Save it using "Save (API Format)" 
        3. Run this method to import it
        """
        workflow = WorkflowTemplates.create_from_ui_export(ui_export_path)
        self.client.save_workflow(workflow, save_as)
        
        # Analyze the workflow
        nodes = WorkflowTemplates.identify_prompt_nodes(workflow)
        print(f"Workflow '{save_as}' imported successfully!")
        print(f"Identified nodes: {json.dumps(nodes, indent=2)}")
        
        return nodes
    
    def generate(self, 
                prompt: str, 
                workflow: str = "default",
                negative: Optional[str] = None,
                seed: int = -1) -> str:
        """
        Simple generation interface for agents
        Returns path to generated image
        """
        if negative is None:
            negative = self.default_negative
        
        if seed == -1:
            import random
            seed = random.randint(0, 0xffffffffffffffff)
        
        # Generate the image
        history = self.client.generate_from_workflow(
            workflow_name=workflow,
            positive_prompt=prompt,
            negative_prompt=negative,
            seed=seed,
            wait=True
        )
        
        # Save and return the image path
        output_path = f"/tmp/comfyui_{uuid.uuid4().hex[:8]}.png"
        saved_files = self.client.save_image_from_history(history, output_path)
        
        return saved_files[0] if saved_files else None


# Example script to set up and use workflows
if __name__ == "__main__":
    # Initialize the agent interface
    agent = SimpleComfyAgent(server="localhost:8188")
    
    # ONE-TIME SETUP: Import a workflow from ComfyUI
    # Uncomment and run once after exporting from ComfyUI:
    # agent.setup_workflow_from_ui(
    #     ui_export_path="./my_workflow_api.json",
    #     save_as="my_custom_workflow"
    # )
    
    # USAGE: Generate images with simple prompts
    image_path = agent.generate(
        prompt="a majestic mountain landscape at sunset, digital art",
        workflow="default",  # Use your saved workflow name
        negative=None,  # Uses default negative prompt
        seed=-1  # Random seed
    )
    
    print(f"Generated image: {image_path}")
    
    # # Advanced usage: Direct workflow manipulation
    # client = ComfyUIWorkflowClient("localhost:8188")
    
    # # Load and modify workflow with specific node IDs
    # workflow = client.load_workflow("default")
    
    # # If you know the exact node IDs from your workflow:
    # workflow = client.update_workflow_advanced(workflow, {
    #     "6": {"text": "beautiful landscape, high quality, 8k"},  # Positive prompt node
    #     "7": {"text": "ugly, blurry"},                          # Negative prompt node  
    #     "3": {"seed": 42, "steps": 25, "cfg": 7.5}             # KSampler node
    # })
    
    # result = client.queue_prompt(workflow)
    # print(f"Queued with ID: {result['prompt_id']}")