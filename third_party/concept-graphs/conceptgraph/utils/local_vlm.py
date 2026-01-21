"""
Local VLM replacements for OpenAI GPT-4V functions
Uses original LLaVA v1.1.3 library compatible with PyTorch 2.0.1 and transformers 4.31.0
"""
import os
import re
import ast
from typing import List, Dict, Optional
from PIL import Image
import torch

# Try importing LLaVA
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.utils import disable_torch_init
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
    LLAVA_AVAILABLE = True
except ImportError as e:
    LLAVA_AVAILABLE = False
    print(f"Warning: LLaVA not available: {e}")

# Try importing transformers for consolidation
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available")

# Default models
DEFAULT_VLM_MODEL = "liuhaotian/llava-v1.5-7b"
# Use TinyLlama which is compatible with transformers 4.31.0 (Mistral requires 4.34+)
DEFAULT_CONSOLIDATOR_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_4bit_quantization_config():
    """Get BitsAndBytes 4-bit quantization config optimized for RTX 4000 series"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # Optimization for RTX 4000 series
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


class LocalVLM:
    """Local Vision-Language Model wrapper using LLaVA v1.1.3 with 4-bit quantization"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda", load_in_4bit: bool = True):
        if not LLAVA_AVAILABLE:
            raise ImportError("LLaVA not available. Run: pip install git+https://github.com/haotian-liu/LLaVA.git@v1.1.3")

        self.model_path = model_path or os.getenv("VLM_MODEL_PATH", DEFAULT_VLM_MODEL)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize LLaVA model with 4-bit quantization"""
        print(f"Loading VLM: {self.model_path}...")
        print(f"4-bit quantization enabled: {self.load_in_4bit}")

        # Disable torch init for faster loading
        disable_torch_init()

        # Get model name from path
        model_name = get_model_name_from_path(self.model_path)
        print(f"Model name: {model_name}")

        # Load model with 4-bit quantization
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=model_name,
            load_4bit=self.load_in_4bit
        )

        print(f"VLM loaded successfully: {self.model_path}")
        print(f"Context length: {self.context_len}")

    def _generate_response(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate a response for an image and prompt"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return ""

        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

            # Build conversation
            conv = conv_templates["v1"].copy()

            # Add image token to prompt
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()

            # Tokenize
            input_ids = tokenizer_image_token(
                prompt_text,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).cuda()

            # Setup stopping criteria
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            # Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

            # Decode response
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:],
                skip_special_tokens=True
            )[0]
            outputs = outputs.strip()

            # Remove stop string if present
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]

            return outputs.strip()

        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def get_obj_captions_from_image(self, image_path: str, label_list: List[str]) -> List[Dict]:
        """
        Generate captions for objects in an image.
        Replaces get_obj_captions_from_image_gpt4v
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []

        captions = []

        # Generate captions for each object
        for label in label_list:
            parts = label.split(": ")
            if len(parts) != 2:
                continue

            obj_id = parts[0].strip()
            obj_name = parts[1].strip()

            prompt = f"Describe the object labeled {obj_id} ({obj_name}) in this image. Be concise (1-2 sentences)."
            caption = self._generate_response(image_path, prompt, max_new_tokens=100)

            captions.append({
                "id": obj_id,
                "name": obj_name,
                "caption": caption
            })

        return captions

    def get_obj_rel_from_image(self, image_path: str, label_list: List[str]) -> List[tuple]:
        """
        Extract spatial relationships between objects.
        Replaces get_obj_rel_from_image_gpt4v
        """
        if not os.path.exists(image_path):
            return []

        label_str = ", ".join(label_list)

        prompt = f"""Analyze the spatial relationships between these labeled objects: {label_str}

Output ONLY a list of tuples showing "on top of" or "under" relationships.
Format: [("1", "on top of", "2"), ("3", "under", "4")]
If no relationships exist, output: []

Only include objects that are physically stacked or placed on each other. Use numeric IDs only."""

        response = self._generate_response(image_path, prompt, max_new_tokens=256)

        relationships = self._extract_tuples(response)
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
                    return [tuple(item) if isinstance(item, (list, tuple)) else item for item in result]
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing tuples: {e}")

        return []


class LocalCaptionConsolidator:
    """Local LLM for consolidating captions with 4-bit quantization"""

    def __init__(self, model_name: Optional[str] = None, device: str = "cuda", load_in_4bit: bool = True):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available")

        self.model_name = model_name or os.getenv("CONSOLIDATOR_MODEL_PATH", DEFAULT_CONSOLIDATOR_MODEL)
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the text model with 4-bit quantization"""
        print(f"Loading consolidator: {self.model_name}...")
        print(f"4-bit quantization enabled: {self.load_in_4bit}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False  # Use slow tokenizer for compatibility with older tokenizers library
        )

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        # Apply 4-bit quantization config optimized for RTX 4000 series
        if self.load_in_4bit:
            quantization_config = get_4bit_quantization_config()
            model_kwargs["quantization_config"] = quantization_config
            print("Using BitsAndBytes 4-bit quantization (nf4, bfloat16 compute, double quant)")
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        print(f"Consolidator loaded successfully: {self.model_name}")

    def consolidate_captions(self, captions: List[Dict]) -> str:
        """
        Consolidate multiple captions into one.
        Replaces consolidate_captions from OpenAI
        """
        if not captions or len(captions) == 0:
            return ""

        # Extract caption texts
        caption_texts = [cap.get('caption', '') for cap in captions if cap.get('caption')]
        if not caption_texts:
            return ""

        if len(caption_texts) == 1:
            return caption_texts[0]

        captions_str = "\n".join([f"- {cap}" for cap in caption_texts])

        # TinyLlama chat format
        prompt = f"""<|system|>
You are a helpful assistant that consolidates multiple descriptions into one clear, concise caption.</s>
<|user|>
Consolidate these captions for the same object into one clear caption:
{captions_str}

Output only the consolidated caption.</s>
<|assistant|>
"""

        try:
            # Use direct generation instead of pipeline for better control
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            input_len = inputs['input_ids'].shape[1]
            consolidated = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

            # Clean up any remaining artifacts
            consolidated = consolidated.replace('</s>', '').strip()
            if '\n' in consolidated:
                consolidated = consolidated.split('\n')[0].strip()

            print(f"Consolidated Caption: {consolidated}")
            return consolidated

        except Exception as e:
            print(f"Error consolidating captions: {e}")

        # Fallback: return the longest caption
        return max(caption_texts, key=len) if caption_texts else ""


# Global instances (lazy loading)
_local_vlm = None
_local_consolidator = None


def get_local_vlm(model_path: Optional[str] = None) -> LocalVLM:
    """Get or create LocalVLM instance"""
    global _local_vlm
    if _local_vlm is None:
        _local_vlm = LocalVLM(model_path=model_path)
    return _local_vlm


def get_local_consolidator(model_name: Optional[str] = None) -> LocalCaptionConsolidator:
    """Get or create LocalCaptionConsolidator instance"""
    global _local_consolidator
    if _local_consolidator is None:
        _local_consolidator = LocalCaptionConsolidator(model_name=model_name)
    return _local_consolidator
