"""
Generate BFS-labeled instruction-tuning data for VLM finetune.

For each training-set transition, compute the BFS-optimal action and save
(upscaled image, prompt, optimal_action) as a JSONL dataset.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from collections import deque
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from renderer import map_desc_to_list

DATA_PATH  = "/data/koe/ECE285-Final/data/transitions.npz"
OUT_DIR    = "/data/koe/ECE285-Final/data/finetune"
NUM_MAPS   = 200
SEED_BASE  = 42
N_SAMPLES  = 25000   # subset — covers ~21% of training data, enough for fine-tune
IMG_SIZE   = 256     # upscale for VLM

ACTION_DELTA = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]

PROMPT = (
    "This is an 8×8 FrozenLake grid game.\n"
    "  - ORANGE circle = your agent (current position)\n"
    "  - GREEN cell with yellow center = GOAL (reach it to win)\n"
    "  - DARK/BLACK cell with dark oval = HOLE (stepping in = game over)\n"
    "  - LIGHT BLUE cells = safe frozen ice\n\n"
    "What is the OPTIMAL next action to navigate toward the goal while avoiding holes?\n"
    "Reply with ONLY ONE WORD: LEFT, DOWN, RIGHT, or UP"
)


def bfs_action(map_desc, state: int):
    """Return BFS-optimal first action, or None if no path."""
    nrows, ncols = len(map_desc), len(map_desc[0])
    goal = next(
        (r * ncols + c for r in range(nrows)
         for c, ch in enumerate(map_desc[r]) if ch == "G"),
        None
    )
    if goal is None or state == goal:
        return None

    queue   = deque([(state, -1)])
    visited = {state}
    while queue:
        cur, fa = queue.popleft()
        row, col = cur // ncols, cur % ncols
        for action, (dr, dc) in ACTION_DELTA.items():
            nr, nc = row + dr, col + dc
            if not (0 <= nr < nrows and 0 <= nc < ncols):
                continue
            nxt = nr * ncols + nc
            if map_desc[nr][nc] == "H" or nxt in visited:
                continue
            visited.add(nxt)
            first = action if fa == -1 else fa
            if nxt == goal:
                return first
            queue.append((nxt, first))
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_dir = f"{OUT_DIR}/images"
    os.makedirs(img_dir, exist_ok=True)

    # ── Regenerate all 200 map descriptors (same RNG as collect_data.py) ──
    rng = np.random.default_rng(SEED_BASE)
    map_descs = []
    for _ in range(NUM_MAPS):
        seed = int(rng.integers(0, 10_000))
        map_descs.append(map_desc_to_list(generate_random_map(size=8, seed=seed)))
    print(f"Regenerated {NUM_MAPS} map descriptors.")

    # ── Load training transitions ──────────────────────────────────────────
    data = np.load(DATA_PATH)
    train_mask = data["train_mask"]
    obs      = data["obs"][train_mask]        # (N, 64, 64, 3)
    states   = data["states"][train_mask].astype(int)
    map_ids  = data["map_ids"][train_mask].astype(int)
    print(f"Training transitions: {len(obs):,}")

    # ── Sample subset ──────────────────────────────────────────────────────
    rng2  = np.random.default_rng(77)
    idxs  = rng2.choice(len(obs), min(N_SAMPLES, len(obs)), replace=False)

    records  = []
    skipped  = 0
    action_counts = [0, 0, 0, 0]

    for i in tqdm(idxs, desc="BFS labeling"):
        state    = states[i]
        map_desc = map_descs[map_ids[i]]
        opt      = bfs_action(map_desc, state)
        if opt is None:
            skipped += 1
            continue

        # Save upscaled image
        img_path = os.path.join(img_dir, f"{i:06d}.png")
        pil = Image.fromarray(obs[i]).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        pil.save(img_path)
        action_counts[opt] += 1

        records.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text",  "text":  PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": ACTION_NAMES[opt],
                },
            ]
        })

    out_path = f"{OUT_DIR}/train.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nGenerated {len(records):,} samples  ({skipped} skipped — no BFS path)")
    print(f"Action distribution: {dict(zip(ACTION_NAMES, action_counts))}")
    print(f"Dataset → {out_path}")
    print(f"Images  → {img_dir}/")


if __name__ == "__main__":
    main()
