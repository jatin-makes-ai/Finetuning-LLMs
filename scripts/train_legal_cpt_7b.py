"""
Stage 1: Continued Pre-Training (CPT) for Qwen2.5-7B on Indian Legal Corpus.

Architecture:
- Base Model: Qwen/Qwen2.5-7B loaded in 8-bit precision (LLM.int8())
- Fine-Tuning Method: Parameter-Efficient Fine-Tuning (PEFT) via LoRA
- LoRA Targets: All 7 attention and MLP linear projections (q, k, v, o, gate, up, down)
- Optimization: AdamW with cosine learning rate schedule, gradient checkpointing
- Training Objective: Standard Causal Language Modeling (Next-Token Prediction) on packed blocks
"""

import os
import math
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

def build_tokenized_dataset(tokenizer, text_path, block_size=1024):
    print(f"[Dataset] Loading raw corpus from: {text_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"[Dataset] Raw text length: {len(raw_text):,} characters")
    
    # Tokenize the entire text without truncation
    print("[Dataset] Tokenizing full corpus...")
    tokens = tokenizer(raw_text, return_attention_mask=False)["input_ids"]
    total_tokens = len(tokens)
    print(f"[Dataset] Total tokens extracted: {total_tokens:,}")

    # Chunk into fixed block_size sequences
    total_blocks = total_tokens // block_size
    print(f"[Dataset] Packing into {total_blocks:,} blocks of length {block_size} (discarding remainder {total_tokens % block_size} tokens)...")
    
    input_ids_list = [
        tokens[i * block_size : (i + 1) * block_size]
        for i in range(total_blocks)
    ]

    dataset = Dataset.from_dict({"input_ids": input_ids_list})
    return dataset

def main():
    model_name_or_path = os.environ.get("MODEL_PATH", "/workspace/models/Qwen2.5-7B")
    corpus_path = os.environ.get("CORPUS_PATH", "/workspace/data/scraped_indian_case_laws_and_books.txt")
    output_dir = os.environ.get("OUTPUT_DIR", "/workspace/models/qwen-7b-legal-cpt-adapter")
    
    block_size = int(os.environ.get("BLOCK_SIZE", "1024"))
    batch_size = int(os.environ.get("BATCH_SIZE", "2"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "8"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-4"))
    epochs = int(os.environ.get("EPOCHS", "1"))
    max_steps = int(os.environ.get("MAX_STEPS", "-1"))

    print("=" * 60)
    print("STAGE 1: QWEN2.5-7B CONTINUED PRE-TRAINING (CPT) - 8-BIT LoRA")
    print("=" * 60)
    print(f"Model Path:         {model_name_or_path}")
    print(f"Corpus Path:        {corpus_path}")
    print(f"Output Adapter:     {output_dir}")
    print(f"Block Size:         {block_size}")
    print(f"Per-Device Batch:   {batch_size}")
    print(f"Gradient Accum:     {grad_accum} (Effective batch size = {batch_size * grad_accum})")
    print(f"Learning Rate:      {learning_rate}")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # 4. Prepare model for k-bit training and configure LoRA
    print("[PEFT] Preparing model for k-bit training and enabling gradient checkpointing...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Build and Pack Dataset
    dataset = build_tokenized_dataset(tokenizer, corpus_path, block_size=block_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=15,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        report_to="none",
        dataloader_num_workers=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # 7. Execute Continued Pre-Training
    print("\n[Training] Starting 8-bit LoRA Continued Pre-Training on RTX 3090...")
    train_result = trainer.train()

    # 8. Log Metrics & Save Adapter
    print("\n" + "=" * 60)
    print("STAGE 1 TRAINING COMPLETE")
    print("=" * 60)
    metrics = train_result.metrics
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    initial_loss = trainer.state.log_history[0].get("loss", "N/A") if trainer.state.log_history else "N/A"
    final_loss = metrics.get("train_loss", "N/A")
    print(f"\nFinal CPT Training Loss: {final_loss}")
    if isinstance(final_loss, float):
        print(f"Domain Perplexity: {math.exp(final_loss):.2f}")

    print(f"\n[Save] Saving trained LoRA adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[Save] Successfully saved adapter weights and tokenizer configuration.")

if __name__ == "__main__":
    main()
