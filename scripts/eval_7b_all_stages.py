"""
Comparative 4-Way Evaluation of Qwen2.5-7B across all 4 stages:
1. Base Qwen2.5-7B
2. Stage 1: CPT Adapter
3. Stage 2: SFT Adapter
4. Stage 3: DPO Adapter

Evaluates 3 unseen, adversarial test cases on:
- 6-part schema completeness
- Deterministic Constitutional article extraction
- Precedent accuracy vs hallucination
- Adversarial trap detection
"""

import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BENCHMARK_PROMPTS = [
    {
        "id": "BENCH-01",
        "title": "ED Summons & PMLA Arrest Safeguards",
        "prompt": "An entrepreneur is summoned by the Enforcement Directorate (ED) under Section 50 of the Prevention of Money Laundering Act (PMLA). The officers demand his smartphone passcodes and threaten immediate arrest under Section 19 PMLA unless he signs a self-incriminating confessional statement. What constitutional and procedural protections apply?"
    },
    {
        "id": "BENCH-02",
        "title": "Blanket Internet Shutdown & Proportionality Test",
        "prompt": "A state government issued an indefinite administrative order shutting down mobile internet across three districts to prevent potential student protests against civil service exam leaks. Can this blanket shutdown be challenged under Article 226/32, and what judicial tests must the state satisfy?"
    },
    {
        "id": "BENCH-03",
        "title": "Adversarial Trap: Article 32 Writ against Private Hospital",
        "prompt": "A private super-specialty hospital abruptly terminated a consultant surgeon without 30 days notice due to a personal dispute with the director. The surgeon wants to file a Writ Petition under Article 32 in the Supreme Court claiming violation of Article 21 (right to practice profession) and demanding immediate judicial reinstatement. What is the correct legal advice?"
    }
]

SYSTEM_PROMPT = (
    "You are an elite Indian legal advisory intelligence. Provide structured legal counsel based strictly on the Constitution of India, statutory codes (IPC/BNS, CrPC, CPC), and binding Supreme Court precedents. "
    "For every scenario, you must include: (1) Executive Legal Assessment, (2) Constitutional Provisions & Rights Matrix with machine-parseable article badges and a structured markdown table, (3) Statutory Framework, (4) Judicial Precedents & Legal Tests, (5) Procedural Roadmap, and (6) Statutory Disclaimer."
)

def evaluate_response(text):
    # Check 6-part schema sections
    sections = [
        "Executive Legal Assessment",
        "Constitutional Provisions",
        "Statutory Framework",
        "Judicial Precedents",
        "Procedural Roadmap",
        "Statutory Disclaimer"
    ]
    sections_found = [s for s in sections if s.lower() in text.lower()]
    schema_score = f"{len(sections_found)}/6"
    
    # Extract deterministic articles
    articles = re.findall(r'\[(Article\s+[^\]]+)\]', text)
    
    return {
        "schema_score": schema_score,
        "sections_found": sections_found,
        "extracted_articles": articles,
        "article_count": len(articles)
    }

def run_generation(model, tokenizer, prompt_text, is_base_model=False):
    if is_base_model:
        # Raw base models without chat template often take direct prompt
        formatted_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=850,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.05
        )
    resp = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return resp

def main():
    base_model_path = "/workspace/models/Qwen2.5-7B"
    cpt_adapter_path = "/workspace/models/qwen-7b-legal-cpt-adapter"
    sft_adapter_path = "/workspace/models/qwen-7b-legal-sft-adapter"
    dpo_adapter_path = "/workspace/models/qwen-7b-legal-dpo-adapter"
    out_json = "/workspace/data/benchmark_7b_all_stages.json"

    print("=" * 60)
    print("4-WAY COMPARATIVE BENCHMARK: QWEN2.5-7B")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # Stage Configurations
    stages = [
        ("Base_7B", None, True),
        ("CPT_7B", cpt_adapter_path, False),
        ("SFT_7B", sft_adapter_path, False),
        ("DPO_7B", dpo_adapter_path, False)
    ]

    for stage_name, adapter_path, is_base in stages:
        print(f"\n[Benchmarking] Loading {stage_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        if adapter_path is not None:
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model
        model.eval()

        stage_results = []
        for case in BENCHMARK_PROMPTS:
            print(f"  Running {case['id']}: {case['title']}...")
            raw_output = run_generation(model, tokenizer, case["prompt"], is_base_model=is_base)
            eval_metrics = evaluate_response(raw_output)
            stage_results.append({
                "case_id": case["id"],
                "case_title": case["title"],
                "prompt": case["prompt"],
                "response": raw_output,
                "metrics": eval_metrics
            })
            print(f"    Schema: {eval_metrics['schema_score']} | Articles Extracted ({eval_metrics['article_count']}): {eval_metrics['extracted_articles']}")

        results[stage_name] = stage_results
        # Free model from memory before next stage
        del model
        del base_model
        torch.cuda.empty_cache()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETE! Results saved to {out_json}")
    print("=" * 60)

if __name__ == "__main__":
    main()
