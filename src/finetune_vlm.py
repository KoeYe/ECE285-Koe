"""
LoRA fine-tune Qwen2-VL-2B-Instruct on BFS-labeled FrozenLake data.

Training objective: given (rendered frame, prompt), output the optimal action word.
Only the LoRA adapter weights are trained; base model is frozen.

Run AFTER gen_finetune_data.py has created data/finetune/train.jsonl.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, TaskType

DATA_PATH  = "/data/koe/ECE285-Final/data/finetune/train.jsonl"
BASE_CACHE = "/data/koe/ECE285-Final/models/qwen2vl"
SAVE_DIR   = "/data/koe/ECE285-Final/models/qwen2vl_ft"
MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE     = "cuda"

BATCH_SIZE  = 4
GRAD_ACCUM  = 8      # effective batch = 32
LR          = 2e-4
EPOCHS      = 3
MAX_SAMPLES = 20000  # use first 20K for reasonable training time
ACTION_NAMES = ["LEFT", "DOWN", "RIGHT", "UP"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class FLDataset(Dataset):
    def __init__(self, path, max_n=MAX_SAMPLES):
        with open(path) as f:
            self.records = [json.loads(l) for l in f]
        if max_n:
            self.records = self.records[:max_n]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


# ── Collate: tokenize a batch, mask prompt tokens in labels ──────────────────

def make_collate(processor):
    def collate(batch):
        full_texts   = []
        prompt_texts = []
        all_images   = []

        for record in batch:
            msgs = record["messages"]
            answer = msgs[-1]["content"]   # e.g. "DOWN"

            # Full conversation (prompt + answer)
            full = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            # Prompt only (to measure length for masking)
            prompt = processor.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True
            )
            full_texts.append(full)
            prompt_texts.append(prompt)

            imgs, _ = process_vision_info(msgs)
            all_images.extend(imgs)

        # Tokenize full sequences
        full_enc = processor(
            text=full_texts,
            images=all_images if all_images else None,
            padding=True,
            return_tensors="pt",
        )

        # Tokenize prompts only (to find boundary)
        prompt_enc = processor(
            text=prompt_texts,
            images=all_images if all_images else None,
            padding=True,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"]        # (B, L_full)
        labels    = input_ids.clone()

        # Mask prompt tokens with -100 (only compute loss on answer tokens)
        for b in range(len(batch)):
            prompt_len = prompt_enc["input_ids"][b].ne(processor.tokenizer.pad_token_id).sum().item()
            labels[b, :prompt_len] = -100

        full_enc["labels"] = labels
        return {k: v for k, v in full_enc.items()
                if k in ("input_ids", "attention_mask", "labels",
                         "pixel_values", "image_grid_thw")}
    return collate


# ── Training loop ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID} …")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        cache_dir=BASE_CACHE,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=BASE_CACHE)
    print("Base model loaded.")

    # ── Attach LoRA ───────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = FLDataset(DATA_PATH)
    print(f"Dataset: {len(dataset):,} samples")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate(processor),
        num_workers=0,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(loader)
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_ok = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            batch = {k: v.to(DEVICE) for k, v in batch.items()
                     if isinstance(v, torch.Tensor)}
            try:
                out  = model(**batch)
                loss = out.loss / GRAD_ACCUM
                loss.backward()
                total_loss += out.loss.item()
                n_ok += 1
            except Exception as e:
                print(f"  step {step} error: {e}")
                optimizer.zero_grad()
                continue

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg = total_loss / max(n_ok, 1)
        print(f"  Epoch {epoch+1} avg loss = {avg:.4f}")

    # ── Save LoRA adapter ─────────────────────────────────────────────────────
    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)
    print(f"\nLoRA adapter saved → {SAVE_DIR}")
    print("Load with: model = PeftModel.from_pretrained(base_model, SAVE_DIR)")


if __name__ == "__main__":
    main()
