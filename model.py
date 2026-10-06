"""Load the base model + fine-tuned LoRA adapter and summarize dialogues (CPU)."""

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "azizjouili4/qwen2.5-0.5b-samsum-lora")
PROMPT = "Summarize the following conversation.\n\n{dialogue}\n\nSummary:"

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
        _model = PeftModel.from_pretrained(base, ADAPTER_REPO)
        _model.eval()
    return _model, _tokenizer


def summarize(dialogue: str, max_new_tokens: int = 64) -> str:
    model, tokenizer = _load()
    prompt = PROMPT.format(dialogue=dialogue.strip())
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()