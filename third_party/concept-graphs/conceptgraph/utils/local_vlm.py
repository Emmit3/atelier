"""
Local VLM replacements for OpenAI GPT-4V functions
"""
import os
import json
import re
import ast
from typing import List, Dict
from PIL import Image
import torch
from pathlib import Path

# Try importing LLaVA
try:
    from conceptgraph.llava.llava_model import LLaVaChat
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    print("Warning: LLaVA not available. Install it or set LLAVA_MODEL_PATH")

# Try importing transformers for consolidation
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available for consolidation")


class LocalVLM:
    """Local Vision-Language Model wrapper using LLaVA"""
    
    def __init__(self, model_path=None, device="cuda", num_gpus=1):
        if not LLAVA_AVAILABLE:
            raise ImportError("LLaVA not available. Please install it.")
        
        self.model_path = model_path or os.getenv("LLAVA_MODEL_PATH")
        if not self.model_path:
            raise ValueError("LLAVA_MODEL_PATH not set and model_path not provided")
        
        self.device = device
        self.num_gpus = num_gpus
        self.chat = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize LLaVA model"""
        print(f"Loading LLaVA from {self.model_path}...")
        self.chat = LLaVaChat(
            model_path=self.model_path,
            conv_mode="multimodal",  # or "v0_mmtag"
            num_gpus=self.num_gpus
        )
        print("LLaVA loaded successfully!")
    
    def get_obj_captions_from_image(self, image_path: str, label_list: List[str]) -> List[Dict]:
        """
        Generate captions for objects in an image using LLaVA.
        Replaces get_obj_captions_from_image_gpt4v
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []
        
        # Load and preprocess image
        image = self.chat.load_image(image_path)
        image_tensor = self.chat.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_features = self.chat.encode_image(image_tensor[None, ...].half().cuda())
        
        captions = []
        
        # Generate caption for each object
        for label in label_list:
            # Extract object ID and name
            parts = label.split(": ")
            if len(parts) != 2:
                continue
            
            obj_id = parts[0].strip()
            obj_name = parts[1].strip()
            
            # Query LLaVA for this object
            query = f"Describe the object labeled {obj_id} ({obj_name}) in this image. Be concise and accurate."
            
            self.chat.reset()
            output = self.chat(query=query, image_features=image_features)
            
            # Clean up the output
            caption = output.strip()
            # Remove common prefixes
            caption = caption.replace("The central object in the image is ", "")
            caption = caption.replace(f"The object labeled {obj_id} is ", "")
            
            captions.append({
                "id": obj_id,
                "name": obj_name,
                "caption": caption
            })
        
        return captions
    
    def get_obj_rel_from_image(self, image_path: str, label_list: List[str]) -> List[tuple]:
        """
        Extract spatial relationships between objects using LLaVA.
        Replaces get_obj_rel_from_image_gpt4v
        """
        if not os.path.exists(image_path):
            return []
        
        # Load and preprocess image
        image = self.chat.load_image(image_path)
        image_tensor = self.chat.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_features = self.chat.encode_image(image_tensor[None, ...].half().cuda())
        
        # Create label list string
        label_str = ", ".join(label_list)
        
        # Query LLaVA for relationships
        query = f"""Analyze the spatial relationships between these objects: {label_str}
        
Output ONLY a list of tuples in this format: [("1", "on top of", "2"), ("3", "under", "2")]
Only include relationships for objects that are physically placed on top of or underneath each other.
If no relationships exist, return an empty list []."""
        
        self.chat.reset()
        output = self.chat(query=query, image_features=image_features)
        
        # Extract list of tuples from output
        relationships = self._extract_tuples(output)
        return relationships
    
    def _extract_tuples(self, text: str) -> List[tuple]:
        """Extract list of tuples from text output"""
        text = text.replace('\n', ' ')
        pattern = r'\[.*?\]'
        match = re.search(pattern, text)
        
        if match:
            list_str = match.group(0)
            try:
                result = ast.literal_eval(list_str)
                if isinstance(result, list):
                    return result
            except (ValueError, SyntaxError):
                pass
        
        return []


class LocalCaptionConsolidator:
    """Local LLM for consolidating captions"""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device="cuda"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available")
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name} for caption consolidation...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with 4-bit quantization to save memory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,  # 4-bit quantization
        )
        
        # Create pipeline for easier generation
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        print(f"{model_name} loaded successfully!")
    
    def consolidate_captions(self, captions: List[Dict]) -> str:
        """
        Consolidate multiple captions into one using local LLM.
        Replaces consolidate_captions from OpenAI
        """
        if not captions or len(captions) == 0:
            return ""
        
        # Extract caption texts
        caption_texts = [cap.get('caption', '') for cap in captions if cap.get('caption')]
        if not caption_texts:
            return ""
        
        # Format prompt
        captions_str = "\n".join([f"- {cap}" for cap in caption_texts])
        
        prompt = f"""<s>[INST] You are an agent specializing in consolidating multiple captions for the same object into a single, clear, and accurate caption.

You will be provided with several captions describing the same object. Your task is to analyze these captions, identify the common elements, remove any noise or outliers, and consolidate them into a single, coherent caption that accurately describes the object.

Here are the captions:
{captions_str}

Please consolidate these into a single, clear caption. Output ONLY the consolidated caption, nothing else. [/INST]"""
        
        try:
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                return_full_text=False,
            )
            
            consolidated = outputs[0]['generated_text'].strip()
            
            # Clean up
            consolidated = consolidated.replace('"', '')
            consolidated = consolidated.replace("'", '')
            
            print(f"Consolidated Caption: {consolidated}")
            return consolidated
            
        except Exception as e:
            print(f"Error consolidating captions: {e}")
            # Fallback: return the longest/most detailed caption
            return max(caption_texts, key=len) if caption_texts else ""


# Global instances (lazy loading)
_local_vlm = None
_local_consolidator = None

def get_local_vlm(model_path=None):
    """Get or create LocalVLM instance"""
    global _local_vlm
    if _local_vlm is None:
        _local_vlm = LocalVLM(model_path=model_path)
    return _local_vlm

def get_local_consolidator(model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    """Get or create LocalCaptionConsolidator instance"""
    global _local_consolidator
    if _local_consolidator is None:
        _local_consolidator = LocalCaptionConsolidator(model_name=model_name)
    return _local_consolidator