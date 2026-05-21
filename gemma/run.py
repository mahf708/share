"""Run google/gemma-4-E4B-it locally on a Perlmutter GPU node.

Usage (inside a GPU allocation):
    python run.py                       # interactive REPL
    python run.py "your prompt here"    # single-shot
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = os.environ.get("GEMMA_MODEL", "google/gemma-4-E4B-it")
MAX_NEW_TOKENS = int(os.environ.get("GEMMA_MAX_NEW_TOKENS", "512"))

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
).eval()


def ask(prompt: str) -> str:
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


if len(sys.argv) > 1:
    print(ask(" ".join(sys.argv[1:])))
else:
    print(f"[gemma] {MODEL_ID} ready on {model.device}. Ctrl-D to exit.")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt:
            print(ask(prompt))
