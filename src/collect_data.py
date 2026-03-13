"""
Collect (o_t, a_t, o_{t+1}) transition tuples from FrozenLake-v1.
Uses 200 random map seeds, random policy, 8x8 non-slippery variant.
Saves to data/transitions.npz with train/val/test splits.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm

from renderer import render_state, map_desc_to_list

# ── Configuration ───────────────────────────────────────────────────────────
NUM_MAPS        = 200       # total distinct map seeds
TRAJ_PER_MAP    = 60        # random trajectories per map
MAX_STEPS       = 40        # max steps per trajectory
GRID_SIZE       = 8
IMG_SIZE        = 64
TRAIN_MAPS      = 160       # 80%
VAL_MAPS        = 20        # 10%
TEST_MAPS       = 20        # 10%
OUT_PATH        = "/data/koe/ECE285-Final/data/transitions.npz"
SEED_BASE       = 42


def collect_transitions(num_maps=NUM_MAPS, traj_per_map=TRAJ_PER_MAP,
                        max_steps=MAX_STEPS, img_size=IMG_SIZE, seed_base=SEED_BASE):
    """Collect all transitions across maps. Returns lists of arrays."""
    all_obs     = []   # (H, W, 3) uint8
    all_next    = []
    all_actions = []   # int8
    all_states  = []   # int16 - raw state index
    all_next_st = []
    all_map_ids = []   # int16 - which map seed
    all_dones   = []

    rng = np.random.default_rng(seed_base)

    for map_id in tqdm(range(num_maps), desc="Collecting maps"):
        map_seed = int(rng.integers(0, 10_000))
        map_desc_str = generate_random_map(size=GRID_SIZE, seed=map_seed)
        map_desc = map_desc_to_list(map_desc_str)

        env = gym.make(
            "FrozenLake-v1",
            desc=map_desc_str,
            is_slippery=False,
            render_mode=None,
        )

        for _ in range(traj_per_map):
            obs_state, _ = env.reset(seed=int(rng.integers(0, 100_000)))
            done = False
            steps = 0

            while not done and steps < max_steps:
                action = int(rng.integers(0, 4))
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                img_t  = render_state(map_desc, obs_state, img_size)
                img_t1 = render_state(map_desc, next_state, img_size)

                all_obs.append(img_t)
                all_next.append(img_t1)
                all_actions.append(action)
                all_states.append(obs_state)
                all_next_st.append(next_state)
                all_map_ids.append(map_id)
                all_dones.append(done)

                obs_state = next_state
                steps += 1

        env.close()

    obs_arr     = np.array(all_obs,     dtype=np.uint8)   # (N, 64, 64, 3)
    next_arr    = np.array(all_next,    dtype=np.uint8)
    actions_arr = np.array(all_actions, dtype=np.int8)
    states_arr  = np.array(all_states,  dtype=np.int16)
    next_st_arr = np.array(all_next_st, dtype=np.int16)
    map_ids_arr = np.array(all_map_ids, dtype=np.int16)
    dones_arr   = np.array(all_dones,   dtype=bool)

    return obs_arr, next_arr, actions_arr, states_arr, next_st_arr, map_ids_arr, dones_arr


def split_by_map(map_ids, n_train=TRAIN_MAPS, n_val=VAL_MAPS, n_test=TEST_MAPS):
    """Return boolean index masks for train/val/test by map id."""
    train_mask = map_ids < n_train
    val_mask   = (map_ids >= n_train) & (map_ids < n_train + n_val)
    test_mask  = map_ids >= n_train + n_val
    return train_mask, val_mask, test_mask


def main():
    print(f"Collecting transitions from {NUM_MAPS} maps...")
    obs, nxt, acts, states, next_st, map_ids, dones = collect_transitions()

    print(f"\nTotal transitions: {len(obs):,}")
    print(f"Image shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Actions distribution: {np.bincount(acts.astype(int))}")

    train_m, val_m, test_m = split_by_map(map_ids)
    print(f"\nSplit sizes:")
    print(f"  Train: {train_m.sum():,}")
    print(f"  Val:   {val_m.sum():,}")
    print(f"  Test:  {test_m.sum():,}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        obs=obs, next_obs=nxt,
        actions=acts, states=states, next_states=next_st,
        map_ids=map_ids, dones=dones,
        train_mask=train_m, val_mask=val_m, test_mask=test_m,
    )
    print(f"\nSaved to {OUT_PATH}")
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
