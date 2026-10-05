"""Before/after evaluation: base model vs LoRA-fine-tuned, scored with ROUGE on SamSum.

Run in Colab after training:
    !python eval.py --adapter adapter --n 100
"""

import argparse

import evaluate
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "Summarize the following conversation.\n\n{dialogue}\n\nSummary:"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default="adapter")
    p.add_argument("--dataset", default="knkarthick/samsum")
    p.add_argument("--n", type=int, default=100, help="number of test examples")
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def generate(model, tokenizer, dialogues, max_new_tokens):
    preds = []
    for d in dialogues:
        prompt = PROMPT.format(dialogue=d)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(text.strip())
    return preds


def main():
    args = parse_args()
    rouge = evaluate.load("rouge")

    test = load_dataset(args.dataset)["test"].select(range(args.n))
    dialogues, refs = test["dialogue"], test["summary"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    base.eval()

    print("Generating with the BASE model...")
    base_preds = generate(base, tokenizer, dialogues, args.max_new_tokens)
    base_scores = rouge.compute(predictions=base_preds, references=refs)

    print("Loading LoRA adapter and generating with the FINE-TUNED model...")
    ft = PeftModel.from_pretrained(base, args.adapter)
    ft.eval()
    ft_preds = generate(ft, tokenizer, dialogues, args.max_new_tokens)
    ft_scores = rouge.compute(predictions=ft_preds, references=refs)

    print(f"\n=== ROUGE on {args.n} SamSum test examples ===")
    print(f"{'metric':10s} {'base':>8s} {'fine-tuned':>12s} {'delta':>8s}")
    for k in ("rouge1", "rouge2", "rougeL"):
        b, f = base_scores[k], ft_scores[k]
        print(f"{k:10s} {b*100:8.2f} {f*100:12.2f} {(f-b)*100:+8.2f}")

    print("\n--- example ---")
    print("DIALOGUE:", dialogues[0][:300], "...")
    print("REFERENCE:", refs[0])
    print("BASE:", base_preds[0])
    print("FINE-TUNED:", ft_preds[0])


if __name__ == "__main__":
    main()