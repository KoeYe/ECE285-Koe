"""
Stage 3: Planning Evaluation

Three policies compared on N_MAPS × N_EPISODES episodes of FrozenLake-v1 8×8:

  1. VLM-ft only        — finetuned Qwen2-VL-2B, zero look-ahead
  2. VLM-ft + WM        — WM predicts next frame for each of 4 actions;
                          VLM evaluates each predicted frame → pick best action
  3. Oracle BFS         — optimal graph-search (upper bound)

Method A design (lookahead consistency):
  At each step, the world model generates 4 one-step predicted frames
  (one per action: L/D/R/U). The finetuned VLM is then asked the SAME
  task it was trained on (POLICY_PROMPT → optimal action word) from each
  predicted frame. An action is filtered out if the VLM from its predicted
  state immediately wants to reverse (e.g., took LEFT → VLM says RIGHT).
  Among surviving actions, the VLM's primary choice from the current frame
  is returned. This avoids train/inference task mismatch (always action
  words, never YES/NO).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from collections import deque
from tqdm import tqdm
import json
from PIL import Image

from model import WorldModel, K, D_LAT
from renderer import render_state, map_desc_to_list

# ── Configuration ─────────────────────────────────────────────────────────────
CKPT_DIR   = "/data/koe/ECE285-Final/checkpoints"
RESULT_DIR = "/data/koe/ECE285-Final/results"
MODEL_BASE  = "/data/koe/ECE285-Final/models/qwen2vl"
MODEL_FT    = "/data/koe/ECE285-Final/models/qwen2vl_ft"
DEVICE     = "cuda"

N_MAPS     = 20
N_EPISODES = 5
MAX_STEPS  = 80
EVAL_SEED  = 1234
IMG_SIZE   = 256    # VLM input resolution

ACTION_DELTA = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

# Prompts
POLICY_PROMPT = (
    "This is an 8×8 FrozenLake grid game.\n"
    "  - ORANGE circle = your agent (current position)\n"
    "  - GREEN cell with yellow center = GOAL (reach it to win)\n"
    "  - DARK/BLACK cell with dark oval = HOLE (stepping in = game over)\n"
    "  - LIGHT BLUE cells = safe frozen ice\n\n"
    "What is the OPTIMAL next action to navigate toward the goal while avoiding holes?\n"
    "Reply with ONLY ONE WORD: LEFT, DOWN, RIGHT, or UP"
)

EVAL_PROMPT = (
    "This is a predicted next game state in FrozenLake 8×8 after taking one action.\n"
    "  - ORANGE circle = agent\n"
    "  - GREEN/YELLOW cell = GOAL (reach it to win)\n"
    "  - DARK/BLACK cell with oval = HOLE (stepping in = instant game over)\n"
    "  - LIGHT BLUE = safe frozen ice\n\n"
    "Is the ORANGE agent on a SAFE cell (light blue or green), NOT on a dark/black hole?\n"
    "Reply with ONLY ONE WORD: YES or NO"
)


# ── Utilities ─────────────────────────────────────────────────────────────────

def load_world_model(tag="default") -> WorldModel:
    ckpt = torch.load(f"{CKPT_DIR}/stage2_{tag}.pth", map_location=DEVICE)
    cfg  = ckpt.get("config", {})
    m    = WorldModel(K=cfg.get("K", K), d_lat=D_LAT).to(DEVICE)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


def obs_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_np).float().permute(2, 0, 1) / 127.5 - 1.0


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) in [-1,1] → (H, W, 3) uint8."""
    arr = (t.clamp(-1, 1).permute(1, 2, 0).cpu().float().numpy() + 1) * 127.5
    return arr.astype(np.uint8)


def find_goal_pos(map_desc) -> tuple:
    for r, row in enumerate(map_desc):
        for c, ch in enumerate(row):
            if ch == "G":
                return (r, c)
    return (7, 7)


# ── Oracle BFS ────────────────────────────────────────────────────────────────

def bfs_action(map_desc, state: int) -> int:
    nrows, ncols = len(map_desc), len(map_desc[0])
    goal = next((r * ncols + c for r in range(nrows)
                 for c, ch in enumerate(map_desc[r]) if ch == "G"), None)
    if goal is None or state == goal:
        return int(np.random.randint(0, 4))

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
    return int(np.random.randint(0, 4))


# ── VLM wrapper ───────────────────────────────────────────────────────────────

class VLMWrapper:
    """
    Thin wrapper around a (possibly LoRA-merged) Qwen2-VL model.
    Supports two query types:
      policy(img)       → action int (0-3)
      score_frame(img)  → float in [0, 1] (YES prob)
    """

    def __init__(self, use_finetuned=True):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        if use_finetuned and os.path.exists(MODEL_FT):
            from peft import PeftModel
            print("[VLM] Loading finetuned model (base + LoRA) …")
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map=DEVICE,
                cache_dir=MODEL_BASE,
            )
            self.model = PeftModel.from_pretrained(base, MODEL_FT)
            self.model = self.model.merge_and_unload()
        else:
            print("[VLM] Loading base model …")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map=DEVICE,
                cache_dir=MODEL_BASE,
            )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", cache_dir=MODEL_BASE
        )
        self._pvi = process_vision_info
        self.model.eval()
        print("[VLM] Ready.")

    def _run(self, img_np: np.ndarray, prompt: str, max_new=5) -> str:
        pil = Image.fromarray(img_np).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text",  "text":  prompt},
        ]}]
        text   = self.processor.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True)
        imgs, vids = self._pvi(msgs)
        inputs = self.processor(text=[text], images=imgs, videos=vids,
                                padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new,
                                     do_sample=False, temperature=None, top_p=None)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip().upper()

    def policy_action(self, obs_img_np: np.ndarray) -> int:
        """Ask VLM for best action from current frame."""
        text = self._run(obs_img_np, POLICY_PROMPT)
        for name, idx in [("LEFT", 0), ("DOWN", 1), ("RIGHT", 2), ("UP", 3)]:
            if name in text:
                return idx
        return int(np.random.randint(0, 4))

    def score_frame(self, pred_img_np: np.ndarray) -> float:
        """Ask VLM if a predicted frame is a good next state. Returns 1=YES, 0=NO."""
        text = self._run(pred_img_np, EVAL_PROMPT)
        return 1.0 if "YES" in text else 0.0


# ── WM one-step predictions ───────────────────────────────────────────────────

def wm_predict_all_actions(wm: WorldModel, obs_tensor: torch.Tensor) -> dict:
    """
    For each of the 4 actions, predict the next frame with the world model.
    Returns {action_int: predicted_img_np (H, W, 3) uint8}.
    """
    obs_batch = obs_tensor.unsqueeze(0).to(DEVICE)   # (1, 3, 64, 64)
    preds = {}
    with torch.no_grad():
        for action in range(4):
            acts = torch.tensor([action], device=DEVICE)
            vis  = wm.encoder(obs_batch)
            atok = wm.action_encoder(acts)
            z    = wm.latent_predictor(vis, atok)
            pred = wm.decoder(obs_batch, z)           # (1, 3, 64, 64)
            preds[action] = tensor_to_img(pred.squeeze(0))
    return preds


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env, map_desc, goal_pos, policy_fn, ep_seed=None) -> tuple:
    state, _ = env.reset(seed=ep_seed)
    done, steps, reward = False, 0, 0.0
    while not done and steps < MAX_STEPS:
        img    = render_state(map_desc, state)
        action = policy_fn(state, img)
        state, reward, terminated, truncated, _ = env.step(action)
        done   = terminated or truncated
        steps += 1
    return bool(reward > 0.5), steps


# ── Test-case generation ──────────────────────────────────────────────────────

def generate_test_cases():
    rng   = np.random.default_rng(EVAL_SEED)
    cases = []
    for _ in range(N_MAPS):
        mseed = int(rng.integers(0, 10_000))
        eps   = [int(rng.integers(0, 100_000)) for _ in range(N_EPISODES)]
        cases.append((mseed, eps))
    return cases


def evaluate(test_cases, policy_fn_factory, label="") -> dict:
    successes, lengths = [], []
    for map_seed, ep_seeds in tqdm(test_cases, desc=f"[{label}]"):
        mds   = generate_random_map(size=8, seed=map_seed)
        mdesc = map_desc_to_list(mds)
        gpos  = find_goal_pos(mdesc)
        pfn   = policy_fn_factory(mdesc, gpos)
        env   = gym.make("FrozenLake-v1", desc=mds,
                         is_slippery=False, render_mode=None)
        for ep_seed in ep_seeds:
            ok, n = run_episode(env, mdesc, gpos, pfn, ep_seed=ep_seed)
            successes.append(ok)
            lengths.append(n)
        env.close()
    n_suc     = sum(successes)
    suc_steps = [l for s, l in zip(successes, lengths) if s]
    return {
        "success_rate":      float(np.mean(successes)),
        "n_success":         n_suc,
        "n_episodes":        len(successes),
        "avg_steps_success": float(np.mean(suc_steps)) if suc_steps else None,
        "avg_steps_all":     float(np.mean(lengths)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Stage 3: Planning Evaluation  device={DEVICE}")
    print(f"N_MAPS={N_MAPS}  N_EPS={N_EPISODES}  MAX_STEPS={MAX_STEPS}")
    print(f"Method A: WM generates 4 predicted frames → VLM lookahead consistency filter\n")

    test_cases = generate_test_cases()
    results    = {}

    # ── 1. Oracle BFS ────────────────────────────────────────────────────────
    def bfs_factory(mdesc, gpos):
        return lambda state, img: bfs_action(mdesc, state)
    results["oracle_bfs"] = evaluate(test_cases, bfs_factory, "Oracle BFS")
    print(f"BFS SR={results['oracle_bfs']['success_rate']*100:.1f}%\n")

    # ── Load models ───────────────────────────────────────────────────────────
    vlm = VLMWrapper(use_finetuned=True)
    wm  = load_world_model("default")

    # ── 2. Finetuned VLM only ─────────────────────────────────────────────────
    def vlm_factory(mdesc, gpos):
        return lambda state, img: vlm.policy_action(img)
    results["vlm_only"] = evaluate(test_cases, vlm_factory, "VLM-ft only")
    print(f"VLM-ft only SR={results['vlm_only']['success_rate']*100:.1f}%\n")

    # ── 3. Finetuned VLM + WM (Method A: lookahead consistency) ──────────────
    # For each action, WM predicts the next frame; VLM is asked (same task as
    # training: POLICY_PROMPT) what it would do FROM that predicted state.
    # A predicted state is "bad" if the VLM from it immediately wants to reverse
    # (e.g., took LEFT → predicted frame → VLM says RIGHT = going back).
    # Filter out reversals, then pick VLM's primary choice among survivors.
    REVERSE = {0: 2, 1: 3, 2: 0, 3: 1}   # LEFT↔RIGHT, DOWN↔UP

    def vlm_wm_factory(mdesc, gpos):
        def policy(state, img):
            obs_t = obs_to_tensor(img)

            # WM: one-step predictions for all 4 actions
            pred_frames = wm_predict_all_actions(wm, obs_t)

            # VLM: ask POLICY_PROMPT from each predicted frame (same task as training)
            future = {}
            for action, pred_img in pred_frames.items():
                future[action] = vlm.policy_action(pred_img)

            # Primary VLM choice from current frame
            primary = vlm.policy_action(img)

            # Filter: drop actions whose predicted state wants to immediately reverse
            good = [a for a in range(4) if future[a] != REVERSE[a]]

            if primary in good:
                return primary
            elif good:
                return good[0]
            else:
                return primary   # all reverse → fallback to primary
        return policy

    results["vlm_wm"] = evaluate(test_cases, vlm_wm_factory, "VLM-ft + WM")
    print(f"VLM-ft+WM SR={results['vlm_wm']['success_rate']*100:.1f}%\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    results["config"] = {
        "n_maps": N_MAPS, "n_eps": N_EPISODES, "max_steps": MAX_STEPS,
        "method": "B", "eval_seed": EVAL_SEED,
    }
    out = f"{RESULT_DIR}/planning_eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    rows = [
        ("VLM-ft only",      results["vlm_only"]),
        ("VLM-ft + WM",      results["vlm_wm"]),
        ("Oracle BFS",       results["oracle_bfs"]),
    ]
    print("\n" + "=" * 58)
    print(f"{'Method':<22} {'Success%':>9} {'Avg Steps (succ)':>16}")
    print("-" * 58)
    for label, r in rows:
        ss = r["avg_steps_success"]
        print(f"{label:<22} {r['success_rate']*100:>8.1f}%"
              f" {(f'{ss:.1f}' if ss else 'N/A'):>16}")
    print("=" * 58)


if __name__ == "__main__":
    main()
