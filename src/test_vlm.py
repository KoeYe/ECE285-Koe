"""Debug VLM raw outputs on FrozenLake frames."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from renderer import render_state, map_desc_to_list
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

CACHE_DIR = "/data/koe/ECE285-Final/models/qwen2vl"
DEVICE    = "cuda"

mds   = generate_random_map(size=8, seed=42)
mdesc = map_desc_to_list(mds)
goal  = [(r, c) for r, row in enumerate(mdesc) for c, ch in enumerate(row) if ch == "G"][0]
print("Goal at", goal)

print("Loading model...")
model     = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16,
    device_map=DEVICE, cache_dir=CACHE_DIR)
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", cache_dir=CACHE_DIR)
model.eval()

# Use a larger upscaled image so VLM can see details
PROMPT = (
    "This is a screenshot of the FrozenLake game on an 8×8 grid.\n"
    "In the image:\n"
    "  - ORANGE filled circle = the player (your agent)\n"
    "  - GREEN cell with yellow center = GOAL (reach here to win)\n"
    "  - DARK NAVY/BLACK cell with dark circle = HOLE (avoid!)\n"
    "  - LIGHT BLUE cells = safe ice\n\n"
    "The player needs to reach the GOAL while avoiding the HOLES.\n"
    "The player starts at the TOP-LEFT and the GOAL is near the BOTTOM-RIGHT.\n\n"
    "What is the single best next action?\n"
    "LEFT moves the player one cell to the left.\n"
    "DOWN moves the player one cell down.\n"
    "RIGHT moves the player one cell to the right.\n"
    "UP moves the player one cell up.\n\n"
    "Answer with ONLY ONE WORD from: LEFT, DOWN, RIGHT, UP"
)

for state in [0, 10, 32, 48]:
    img_np = render_state(mdesc, state)
    # Upscale 4× so VLM can see details
    pil = Image.fromarray(img_np).resize((256, 256), Image.NEAREST)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil},
            {"type": "text",  "text":  PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_inp, vid_inp = process_vision_info(messages)
    inputs = processor(text=[text], images=img_inp, videos=vid_inp,
                       padding=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                             temperature=None, top_p=None)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    row, col = state // 8, state % 8
    print(f"\nState {state:2d} (r={row},c={col}):")
    print(f"  Raw output: '{raw}'")
