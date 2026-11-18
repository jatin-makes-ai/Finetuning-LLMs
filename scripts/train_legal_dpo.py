#!/usr/bin/env python3
"""
train_legal_dpo.py

Stage 3: Direct Preference Optimization (DPO) for Indian Legal AI Assistant.
- Loads Stage 2 SFT merged model /workspace/models/qwen-0.5b-legal-complete.
- Attaches fresh LoRA adapter (r=16, alpha=32) for preference alignment.
- Loads 20 curated (prompt, chosen, rejected) legal pairs.
- Trains with trl.DPOTrainer using beta=0.1 (implicit KL penalty).
- Disables adapter for zero-overhead reference model pi_ref.
- Merges and saves the final aligned model to /workspace/models/qwen-0.5b-legal-dpo-complete.
"""

import json
import os
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import DPOConfig, DPOTrainer

SFT_MODEL_DIR = "/workspace/models/qwen-0.5b-legal-complete"
DPO_DATA_PATH = "/workspace/data/legal_dpo_dataset.jsonl"
DPO_ADAPTER_OUTPUT = "/workspace/adapters/qwen-0.5b-legal-dpo"
FINAL_DPO_MODEL_OUTPUT = "/workspace/models/qwen-0.5b-legal-dpo-complete"
TRAINING_OUTPUT_DIR = "/workspace/dpo_output"


def load_dpo_dataset(file_path: str, tokenizer: AutoTokenizer) -> Dataset:
    raw_samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))

    processed_samples = []
    for item in raw_samples:
        prompt_str = tokenizer.apply_chat_template(
            item["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        chosen_str = item["chosen"] + tokenizer.eos_token
        rejected_str = item["rejected"] + tokenizer.eos_token

        processed_samples.append({
            "prompt": prompt_str,
            "chosen": chosen_str,
            "rejected": rejected_str,
        })

    return Dataset.from_list(processed_samples)


def main():
    print(f"=== Stage 3: Direct Preference Optimization (DPO) ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Stage 2 SFT Merged Model
    print(f"\n[1/5] Loading Stage 2 SFT Merged Model: {SFT_MODEL_DIR}...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_DIR,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. Attach LoRA Adapter for DPO Alignment
    print(f"\n[2/5] Attaching LoRA Adapter for DPO Alignment...")
    dpo_lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, dpo_lora_config)
    model.print_trainable_parameters()

    # 4. Load & Process Preference Dataset
    print(f"\n[3/5] Loading Preference Dataset: {DPO_DATA_PATH}...")
    train_dataset = load_dpo_dataset(DPO_DATA_PATH, tokenizer)
    print(f"Loaded {len(train_dataset)} preference pairs.")

    # 5. DPO Configuration
    print(f"\n[4/5] Initializing DPOConfig (beta=0.1, lr=5e-5)...")
    dpo_config = DPOConfig(
        output_dir=TRAINING_OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=3,
        logging_steps=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        save_strategy="no",
        report_to="none",
        beta=0.1,
        max_length=1024,
        dataloader_drop_last=False,
    )

    # Note: When model is PeftModel and ref_model=None, DPOTrainer automatically
    # disables the LoRA adapter to compute the reference model pi_ref forward pass.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # 6. Run Training
    print("\n[5/5] Executing DPO Training...")
    train_result = trainer.train()
    print(f"\nDPO Training Completed! Final Loss: {train_result.training_loss:.4f}")

    # 7. Save DPO Adapter
    os.makedirs(DPO_ADAPTER_OUTPUT, exist_ok=True)
    trainer.model.save_pretrained(DPO_ADAPTER_OUTPUT)
    tokenizer.save_pretrained(DPO_ADAPTER_OUTPUT)
    print(f"Saved DPO LoRA adapter to: {DPO_ADAPTER_OUTPUT}")

    # 8. Merge and Save Standalone DPO Model
    print(f"\nMerging DPO weights into final standalone checkpoint...")
    final_merged = trainer.model.merge_and_unload()
    os.makedirs(FINAL_DPO_MODEL_OUTPUT, exist_ok=True)
    final_merged.save_pretrained(FINAL_DPO_MODEL_OUTPUT)
    tokenizer.save_pretrained(FINAL_DPO_MODEL_OUTPUT)
    print(f"Saved complete DPO-aligned model to: {FINAL_DPO_MODEL_OUTPUT}")
    print("\n=== Stage 3 DPO Pipeline Complete! ===")


if __name__ == "__main__":
    main()
