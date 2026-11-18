#!/usr/bin/env python3
"""
eval_all_stages.py

Comprehensive 3-Way Comparative Benchmark:
Stage 0: Raw Base Model (Qwen/Qwen2.5-0.5B)
Stage 1: Base + Legal CPT Adapter (/workspace/adapters/qwen-0.5b-legal-cpt)
Stage 2: Complete Legal AI Assistant (/workspace/models/qwen-0.5b-legal-complete)
"""

import json
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CPT_ADAPTER_DIR = "/workspace/adapters/qwen-0.5b-legal-cpt"
FINAL_MODEL_DIR = "/workspace/models/qwen-0.5b-legal-complete"
OUTPUT_EVAL_FILE = "/workspace/stage2_comparative_eval.json"

TEST_PROMPTS = [
    {
        "id": "arrest_rights",
        "question": "What are the immediate legal rights of an Indian citizen arrested without a warrant under Section 41 CrPC?",
    },
    {
        "id": "writ_jurisdiction",
        "question": "Explain the essential differences between approaching the Supreme Court under Article 32 and the High Court under Article 226.",
    },
    {
        "id": "basic_structure",
        "question": "Can Parliament amend any part of the Constitution under Article 368? What is the Basic Structure Doctrine?",
    },
    {
        "id": "cheque_bounce",
        "question": "A cheque of Rs 5,00,000 bounced due to insufficient funds. What is the exact procedure and limitation timeline under Section 138 of the Negotiable Instruments Act?",
    }
]

SYSTEM_PROMPT = (
    "You are an expert Indian Legal AI Assistant. You provide precise, authoritative legal counsel "
    "grounded in the Constitution of India, central statutory codes, and landmark Supreme Court precedents. "
    "Always structure your analysis with clarity, citing applicable statutory provisions, relevant case law, "
    "and concrete procedural remedies."
)


def generate_response(model, tokenizer, prompt_text: str, is_chat: bool = True, max_new_tokens: int = 512) -> str:
    if is_chat and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = f"User: {prompt_text}\nAssistant:"

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main():
    print("=== Master 3-Way Legal Adaptation Benchmark ===")
    results = {}
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------------------------------
    # Stage 0: Raw Base Model
    # ----------------------------------------------------
    print("\n[Stage 0/3] Evaluating Raw Base Model: Qwen/Qwen2.5-0.5B...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    stage0_results = {}
    for p in TEST_PROMPTS:
        print(f"  Testing Prompt: {p['id']}...")
        stage0_results[p["id"]] = generate_response(base_model, tokenizer, p["question"], is_chat=True)

    results["stage0_raw_base"] = stage0_results

    # ----------------------------------------------------
    # Stage 1: Base + CPT Adapter
    # ----------------------------------------------------
    print("\n[Stage 1/3] Evaluating Base + Stage 1 Legal CPT Adapter...")
    cpt_model = PeftModel.from_pretrained(base_model, CPT_ADAPTER_DIR)

    stage1_results = {}
    for p in TEST_PROMPTS:
        print(f"  Testing Prompt: {p['id']}...")
        stage1_results[p["id"]] = generate_response(cpt_model, tokenizer, p["question"], is_chat=True)

    results["stage1_cpt"] = stage1_results

    # Free memory
    del cpt_model
    del base_model
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # Stage 2: Merged Stage 2 Legal AI Model
    # ----------------------------------------------------
    print(f"\n[Stage 2/3] Evaluating Complete Merged Model from {FINAL_MODEL_DIR}...")
    final_model = AutoModelForCausalLM.from_pretrained(
        FINAL_MODEL_DIR,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    stage2_results = {}
    for p in TEST_PROMPTS:
        print(f"  Testing Prompt: {p['id']}...")
        stage2_results[p["id"]] = generate_response(final_model, tokenizer, p["question"], is_chat=True)

    results["stage2_sft"] = stage2_results

    # Save output
    with open(OUTPUT_EVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nComparative results saved to {OUTPUT_EVAL_FILE}")


if __name__ == "__main__":
    main()
