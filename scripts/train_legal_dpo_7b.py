"""
Stage 3: Direct Preference Optimization (DPO) for Qwen2.5-7B on Indian Legal Preference Dataset.

Architecture:
- Base Model: Qwen/Qwen2.5-7B loaded in 8-bit precision (LLM.int8())
- Policy Model (pi_theta): Warm-started from Stage 2 SFT adapter
- Reference Model (pi_ref): Evaluated with zero extra VRAM by disabling LoRA adapter
- Loss: Bradley-Terry preference loss with beta = 0.1
- Metrics: Tracks Policy Loss, Chosen Reward, Rejected Reward, Margin, and Reward Accuracy
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    PeftModel,
    prepare_model_for_kbit_training
)
from trl import DPOConfig, DPOTrainer

class DPOProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs.get("loss", 0.0)
            margin = logs.get("rewards/margins", 0.0)
            accuracy = logs.get("rewards/accuracies", 0.0)
            r_chosen = logs.get("rewards/chosen", 0.0)
            r_rejected = logs.get("rewards/rejected", 0.0)
            lr = logs.get("learning_rate", 0.0)
            print(
                f">>> [DPO PROGRESS] Step {state.global_step:2d}/{state.max_steps} | "
                f"Loss: {loss:.4f} | Margin: {margin:+.4f} | Acc: {accuracy*100:.1f}% | "
                f"R_w: {r_chosen:+.2f} | R_l: {r_rejected:+.2f} | LR: {lr:.2e}",
                flush=True
            )

def load_and_format_dpo_dataset(jsonl_path, tokenizer):
    print(f"[Dataset] Loading DPO preference dataset from {jsonl_path}...")
    system_prompt = (
        "You are an elite Indian legal advisory intelligence. Provide structured legal counsel based strictly on the Constitution of India, statutory codes (IPC/BNS, CrPC, CPC), and binding Supreme Court precedents. "
        "For every scenario, you must include: (1) Executive Legal Assessment, (2) Constitutional Provisions & Rights Matrix with machine-parseable article badges and a structured markdown table, (3) Statutory Framework, (4) Judicial Precedents & Legal Tests, (5) Procedural Roadmap, and (6) Statutory Disclaimer."
    )

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    formatted_records = []
    for item in records:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["prompt"]}
        ]
        prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_records.append({
            "prompt": prompt_str,
            "chosen": item["chosen"] + tokenizer.eos_token,
            "rejected": item["rejected"] + tokenizer.eos_token
        })

    print(f"[Dataset] Successfully formatted {len(formatted_records)} preference pairs.")
    return Dataset.from_list(formatted_records)

def main():
    model_name_or_path = os.environ.get("MODEL_PATH", "/workspace/models/Qwen2.5-7B")
    sft_adapter_path = os.environ.get("SFT_ADAPTER_PATH", "/workspace/models/qwen-7b-legal-sft-adapter")
    data_path = os.environ.get("DATA_PATH", "/workspace/data/legal_dpo_7b_dataset.jsonl")
    output_dir = os.environ.get("OUTPUT_DIR", "/workspace/models/qwen-7b-legal-dpo-adapter")

    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "5e-6"))
    epochs = int(os.environ.get("EPOCHS", "2"))
    beta = float(os.environ.get("BETA", "0.1"))

    print("=" * 60)
    print("STAGE 3: QWEN2.5-7B DIRECT PREFERENCE OPTIMIZATION (DPO) - 8-BIT")
    print("=" * 60)
    print(f"Base Model:         {model_name_or_path}")
    print(f"SFT Policy Adapter: {sft_adapter_path}")
    print(f"DPO Dataset:        {data_path}")
    print(f"Output Adapter:     {output_dir}")
    print(f"Per-Device Batch:   {batch_size}")
    print(f"Gradient Accum:     {grad_accum} (Effective batch size = {batch_size * grad_accum})")
    print(f"Learning Rate:      {learning_rate}")
    print(f"Beta (Penalty):     {beta}")
    print(f"Epochs:             {epochs}")
    print("=" * 60)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        sft_adapter_path,
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

    # 4. Prepare for k-bit training and load SFT Adapter as policy pi_theta
    print("[PEFT] Preparing base model for k-bit training with gradient checkpointing...")
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    print(f"[PEFT] Loading SFT adapter: {sft_adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_path,
        is_trainable=True
    )
    model.print_trainable_parameters()

    # 5. Format DPO Dataset
    dataset = load_and_format_dpo_dataset(data_path, tokenizer)

    # 6. DPO Configuration
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=3,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        bf16=True,
        beta=beta,
        max_length=2048,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Automatically evaluated via model.disable_adapter()
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[DPOProgressCallback()]
    )

    # 7. Execute DPO Training
    print("\n[Training] Starting Stage 3 DPO Training on RTX 3090...")
    train_result = dpo_trainer.train()

    # 8. Log Metrics & Save DPO Adapter
    print("\n" + "=" * 60)
    print("STAGE 3 DPO ALIGNMENT COMPLETE")
    print("=" * 60)
    for k, v in train_result.metrics.items():
        print(f"  {k}: {v}")

    print(f"\n[Save] Persisting aligned DPO adapter to {output_dir}...")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[Save] Successfully saved DPO aligned adapter weights and tokenizer configuration.")

if __name__ == "__main__":
    main()
