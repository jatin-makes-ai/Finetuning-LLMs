#!/usr/bin/env python3
"""
train_legal_sft.py

Stage 2: Supervised Fine-Tuning (SFT) for Indian Legal AI Assistant.
- Loads raw base model Qwen/Qwen2.5-0.5B.
- Merges Stage 1 CPT LoRA weights into base weights (domain statutory grounding).
- Applies a fresh LoRA adapter (r=16, alpha=32) for instruction following.
- Applies Prompt-Loss Masking (labels=-100 for system & user prompt tokens).
- Trains using HuggingFace Trainer with bf16/fp16 on NVIDIA RTX 3090.
- Saves both the SFT LoRA adapter and the fully merged standalone model.
"""

import json
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CPT_ADAPTER_DIR = "/workspace/adapters/qwen-0.5b-legal-cpt"
SFT_DATA_PATH = "/workspace/data/legal_sft_dataset.jsonl"
SFT_ADAPTER_OUTPUT = "/workspace/adapters/qwen-0.5b-legal-sft"
MERGED_MODEL_OUTPUT = "/workspace/models/qwen-0.5b-legal-complete"
TRAINING_OUTPUT_DIR = "/workspace/sft_output"


def load_sft_dataset(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_chatml_and_mask(sample: Dict, tokenizer: AutoTokenizer, max_length: int = 1024):
    """
    Formats the conversation with ChatML and masks system/user prompt tokens with -100
    so loss is strictly calculated on the assistant response.
    """
    messages = sample["messages"]
    
    # 1. Full text tokenization
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    labels = input_ids.clone()

    # 2. Extract prompt (all messages except final assistant response)
    prompt_messages = messages[:-1]
    # For Qwen ChatML, appending generation prompt gives: <|im_start|>assistant\n
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_encoded = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )
    prompt_len = prompt_encoded["input_ids"].shape[1]

    # 3. Mask prompt tokens with -100
    labels[:prompt_len] = -100

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
    }


@dataclass
class SFTDataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tensor_features = []
        for f in features:
            tensor_features.append({
                "input_ids": torch.as_tensor(f["input_ids"], dtype=torch.long),
                "attention_mask": torch.as_tensor(f["attention_mask"], dtype=torch.long),
                "labels": torch.as_tensor(f["labels"], dtype=torch.long),
            })

        max_len = max(len(tf["input_ids"]) for tf in tensor_features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for tf in tensor_features:
            input_ids = tf["input_ids"]
            att_mask = tf["attention_mask"]
            labels = tf["labels"]

            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                padded_input_ids.append(torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
                padded_attention_mask.append(torch.cat([att_mask, torch.zeros(pad_len, dtype=torch.long)]))
                padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)]))
            else:
                padded_input_ids.append(input_ids)
                padded_attention_mask.append(att_mask)
                padded_labels.append(labels)

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }


def main():
    print(f"=== Stage 2: Supervised Fine-Tuning (SFT) ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model & Fuse CPT Adapter
    print(f"\n[1/5] Loading Base Model: {BASE_MODEL_NAME}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.exists(CPT_ADAPTER_DIR):
        print(f"[2/5] Fusing Stage 1 CPT LoRA weights from {CPT_ADAPTER_DIR}...")
        cpt_model = PeftModel.from_pretrained(base_model, CPT_ADAPTER_DIR)
        model = cpt_model.merge_and_unload()
        print("Successfully merged CPT adapter into base model weights!")
    else:
        print(f"[WARN] CPT adapter not found at {CPT_ADAPTER_DIR}, continuing with raw base...")
        model = base_model

    # 3. Attach Fresh LoRA Adapter for SFT
    print(f"\n[3/5] Attaching Stage 2 SFT LoRA Adapter...")
    sft_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, sft_lora_config)
    model.print_trainable_parameters()

    # 4. Prepare SFT Dataset with Prompt-Loss Masking
    print(f"\n[4/5] Loading & Tokenizing SFT Dataset: {SFT_DATA_PATH}...")
    raw_samples = load_sft_dataset(SFT_DATA_PATH)
    processed_samples = [format_chatml_and_mask(s, tokenizer) for s in raw_samples]
    
    # Calculate token stats
    total_tokens = sum(len(s["input_ids"]) for s in processed_samples)
    loss_tokens = sum(sum(1 for lbl in s["labels"] if lbl != -100) for s in processed_samples)
    print(f"Total dataset samples: {len(processed_samples)}")
    print(f"Total tokens: {total_tokens:,} | Assistant response tokens (loss computed): {loss_tokens:,} ({loss_tokens/total_tokens*100:.1f}%)")

    dataset = Dataset.from_list(processed_samples)
    data_collator = SFTDataCollator(tokenizer=tokenizer)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=TRAINING_OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        save_strategy="no",
        report_to="none",
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\n[5/5] Starting SFT LoRA Training...")
    train_result = trainer.train()
    print(f"\nTraining completed! Final Loss: {train_result.training_loss:.4f}")

    # 6. Save SFT Adapter
    os.makedirs(SFT_ADAPTER_OUTPUT, exist_ok=True)
    model.save_pretrained(SFT_ADAPTER_OUTPUT)
    tokenizer.save_pretrained(SFT_ADAPTER_OUTPUT)
    print(f"Saved Stage 2 SFT LoRA adapter to: {SFT_ADAPTER_OUTPUT}")

    # 7. Merge and Save Final Standalone Model
    print(f"\nMerging Stage 2 weights into standalone checkpoint...")
    final_merged = model.merge_and_unload()
    os.makedirs(MERGED_MODEL_OUTPUT, exist_ok=True)
    final_merged.save_pretrained(MERGED_MODEL_OUTPUT)
    tokenizer.save_pretrained(MERGED_MODEL_OUTPUT)
    print(f"Saved complete merged standalone legal model to: {MERGED_MODEL_OUTPUT}")
    print("\n=== Stage 2 SFT Pipeline Complete! ===")


if __name__ == "__main__":
    main()
