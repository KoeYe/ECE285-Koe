"""
VLM policy wrapper for Qwen2-VL-2B-Instruct on FrozenLake.

Usage:
    vlm = VLMPolicy()
    action = vlm.get_action(obs_img_np)          # single best action (0-3)
    probs  = vlm.get_action_distribution(obs_img_np)  # biased prob array
"""

import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

VLM_IMG_SIZE = 256   # upscale 64→256 so VLM can resolve cell details

MODEL_ID  = "Qwen/Qwen2-VL-2B-Instruct"
CACHE_DIR = "/data/koe/ECE285-Final/models/qwen2vl"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# Map VLM text output → action int
_ACTION_MAP = {
    "LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3,
}

# Prompt: describe image contents and ask for one-word action
_USER_PROMPT = (
    "This is an 8x8 FrozenLake grid game.\n\n"
    "In the image:\n"
    "  - ORANGE circle = your agent (current position)\n"
    "  - GREEN cell with yellow center = GOAL (reach it to win)\n"
    "  - DARK/BLACK cell with dark oval = HOLE (avoid! stepping in = game over)\n"
    "  - LIGHT BLUE cells = safe frozen ice\n\n"
    "The goal is near the BOTTOM-RIGHT corner.\n"
    "Navigate the orange agent to the green goal while avoiding the dark holes.\n\n"
    "Available actions:\n"
    "  LEFT  – move one cell to the left\n"
    "  DOWN  – move one cell down\n"
    "  RIGHT – move one cell to the right\n"
    "  UP    – move one cell up\n"
    "(Moving into a wall keeps you in place.)\n\n"
    "What is the single best next action? Reply with ONLY ONE WORD: LEFT, DOWN, RIGHT, or UP"
)


class VLMPolicy:
    """Thin wrapper around Qwen2-VL-2B-Instruct for FrozenLake action selection."""

    def __init__(self, model_id: str = MODEL_ID,
                 cache_dir: str = CACHE_DIR,
                 device: str = DEVICE):
        import os
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[VLMPolicy] Loading {model_id} …", flush=True)
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir=cache_dir,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model.eval()
        print("[VLMPolicy] Ready.", flush=True)

    # ── internal ──────────────────────────────────────────────────────────────

    def _build_inputs(self, pil_img: Image.Image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text":  _USER_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    def _decode(self, inputs) -> str:
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, gen_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip().upper()

    # ── public API ────────────────────────────────────────────────────────────

    def get_action(self, obs_img_np: np.ndarray) -> int:
        """
        obs_img_np: (H, W, 3) uint8 RGB
        Returns best action int (0=LEFT 1=DOWN 2=RIGHT 3=UP).
        Falls back to random if parsing fails.
        """
        # Upscale so VLM can resolve individual cell details
        pil = Image.fromarray(obs_img_np).resize(
            (VLM_IMG_SIZE, VLM_IMG_SIZE), Image.NEAREST
        )
        inputs = self._build_inputs(pil)
        text = self._decode(inputs)

        for name, idx in _ACTION_MAP.items():
            if name in text:
                return idx

        # fallback
        return int(np.random.randint(0, 4))

    def get_action_distribution(self, obs_img_np: np.ndarray,
                                bias: float = 0.7) -> np.ndarray:
        """
        Returns a (4,) probability array.
        VLM-preferred action receives `bias`; remaining 3 share (1 − bias).
        Used to guide MPC sampling toward VLM-preferred first actions.
        """
        best  = self.get_action(obs_img_np)
        probs = np.full(4, (1.0 - bias) / 3.0)
        probs[best] = bias
        return probs
