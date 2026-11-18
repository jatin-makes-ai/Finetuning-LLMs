import os
import time
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
CORPUS_PATH = "/workspace/data/legal_corpus.txt"
ADAPTER_PATH = "/workspace/adapters/qwen-0.5b-legal-cpt"
BLOCK_SIZE = 512

print("="*60)
print("      STAGE 1: DOMAIN-ADAPTIVE CONTINUED PRE-TRAINING (CPT)")
print("="*60)
print(f"  Base Model:      {MODEL_ID}")
print(f"  Corpus:          {CORPUS_PATH}")
print(f"  Block Size:      {BLOCK_SIZE} tokens")
print(f"  Output Adapter:  {ADAPTER_PATH}")
print("="*60 + "\n")

# 1. Load Tokenizer & Base Model
print("[Step 1/5] Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

# 2. Configure LoRA for Continued Pre-Training
print("[Step 2/5] Setting up LoRA configuration (r=16, alpha=32)...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Prepare Tokenized Continuous Chunks
print("\n[Step 3/5] Tokenizing and packing legal corpus into continuous sequences...")
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize full text as a continuous token stream
tokenized = tokenizer(text, truncation=False, return_attention_mask=False)["input_ids"]

# Chunk into blocks of BLOCK_SIZE
total_length = len(tokenized)
chunks = [tokenized[i : i + BLOCK_SIZE] for i in range(0, total_length - BLOCK_SIZE + 1, BLOCK_SIZE)]
print(f"           Extracted {len(chunks)} training blocks of {BLOCK_SIZE} tokens each ({len(chunks)*BLOCK_SIZE:,} total tokens).")

train_dataset = Dataset.from_dict({"input_ids": chunks})

# 4. Data Collator for Causal LM (Auto-regressive labels)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 5. Training Arguments & Trainer
print("\n[Step 4/5] Starting CPT training on NVIDIA RTX 3090...")
training_args = TrainingArguments(
    output_dir="/workspace/cpt_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=5,
    logging_steps=5,
    bf16=True,
    report_to="none",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

start_time = time.time()
trainer.train()
train_time = time.time() - start_time
print(f"\n[Step 5/5] CPT Training completed in {train_time:.1f}s!")

# Save the trained CPT adapter
os.makedirs(ADAPTER_PATH, exist_ok=True)
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"           Legal CPT adapter successfully saved to {ADAPTER_PATH}!\n")
