"""
Stage 2: Supervised Fine-Tuning (SFT) for Qwen2.5-7B on Structured Legal Advisory Dataset.

Architecture:
- Base Model: Qwen/Qwen2.5-7B loaded in 8-bit precision (LLM.int8())
- Warm-Start: Initialized directly from Stage 1 CPT LoRA Adapter
- Loss Masking: Strict Prompt Masking (labels = -100 on system & user tokens; gradients flow solely on the 6-part legal opinion completion)
- Target Output: Enforces invariant 6-part schema including the deterministic Constitutional Provisions Matrix
"""

import os
import math
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import (
    PeftModel,
    prepare_model_for_kbit_training
)

class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs.get("loss")
            lr = logs.get("learning_rate", 0.0)
            epoch = logs.get("epoch", 0.0)
            print(f">>> [SFT PROGRESS] Step {state.global_step:2d}/{state.max_steps} | Loss: {loss:.4f} | LR: {lr:.3e} | Epoch: {epoch:.2f}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        print(f">>> [SFT CHECKPOINT] Checkpoint saved at step {state.global_step}!", flush=True)

class SFTDataCollator:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in features:
            messages = item["messages"]
            prompt_msgs = messages[:-1]
            response_msg = messages[-1]

            # Format prompt and response using Qwen chat template
            prompt_str = self.tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True
            )
            response_str = response_msg["content"] + self.tokenizer.eos_token

            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            response_ids = self.tokenizer.encode(response_str, add_special_tokens=False)

            input_ids = prompt_ids + response_ids
            # Prompt masking: labels = -100 for all system and user prompt tokens
            labels = [-100] * len(prompt_ids) + response_ids

            # Truncate if exceeds max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append([1] * len(input_ids))

        # Dynamic padding across the batch
        max_batch_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        pad_id = self.tokenizer.pad_token_id

        for input_ids, mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_batch_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }

def load_sft_dataset(jsonl_path):
    print(f"[Dataset] Loading SFT dataset from {jsonl_path}...")
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"[Dataset] Successfully loaded {len(records)} instruction cases.")
    return Dataset.from_list(records)

def main():
    model_name_or_path = os.environ.get("MODEL_PATH", "/workspace/models/Qwen2.5-7B")
    cpt_adapter_path = os.environ.get("CPT_ADAPTER_PATH", "/workspace/models/qwen-7b-legal-cpt-adapter")
    data_path = os.environ.get("DATA_PATH", "/workspace/data/legal_sft_7b_dataset.jsonl")
    output_dir = os.environ.get("OUTPUT_DIR", "/workspace/models/qwen-7b-legal-sft-adapter")

    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))
    epochs = int(os.environ.get("EPOCHS", "3"))

    print("=" * 60)
    print("STAGE 2: QWEN2.5-7B SUPERVISED FINE-TUNING (SFT) - 8-BIT LoRA")
    print("=" * 60)
    print(f"Base Model:         {model_name_or_path}")
    print(f"Warm-Start Adapter: {cpt_adapter_path}")
    print(f"SFT Dataset:        {data_path}")
    print(f"Output Adapter:     {output_dir}")
    print(f"Per-Device Batch:   {batch_size}")
    print(f"Gradient Accum:     {grad_accum} (Effective batch size = {batch_size * grad_accum})")
    print(f"Learning Rate:      {learning_rate}")
    print(f"Epochs:             {epochs}")
    print("=" * 60)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Configure 8-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    # 3. Load Base Model in 8-bit
    print("\n[Model] Loading base model weights in 8-bit precision...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # 4. Prepare for k-bit training and load CPT LoRA Adapter
    print("[PEFT] Preparing model for k-bit training with gradient checkpointing...")
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    print(f"[PEFT] Warm-starting from CPT adapter: {cpt_adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        cpt_adapter_path,
        is_trainable=True
    )
    model.print_trainable_parameters()

    # 5. Load Dataset and Collator
    raw_dataset = load_sft_dataset(data_path)
    data_collator = SFTDataCollator(tokenizer=tokenizer, max_length=2048)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=5,
        lr_scheduler_type="cosine",
        logging_steps=2,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=15,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        report_to="none",
        dataloader_num_workers=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_dataset,
        data_collator=data_collator,
        callbacks=[ProgressCallback()]
    )

    # 7. Execute SFT Training
    print("\n[Training] Starting Stage 2 SFT Training on RTX 3090...")
    train_result = trainer.train()

    # 8. Log Metrics & Save SFT Adapter
    print("\n" + "=" * 60)
    print("STAGE 2 SFT TRAINING COMPLETE")
    print("=" * 60)
    for k, v in train_result.metrics.items():
        print(f"  {k}: {v}")

    print(f"\n[Save] Persisting trained SFT adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[Save] Successfully saved SFT adapter weights and tokenizer configuration.")

if __name__ == "__main__":
    main()
