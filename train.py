"""LoRA fine-tuning of a small LLM for dialogue summarization (SamSum).

Runs on a free Colab T4. Logs to Weights & Biases and saves the LoRA adapter.

Example (in Colab):
    !python train.py --epochs 1 --max-steps 400 --wandb-project dialogue-summarizer
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

PROMPT = "Summarize the following conversation.\n\n{dialogue}\n\nSummary:"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="knkarthick/samsum")
    p.add_argument("--output-dir", default="adapter")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="-1 = full epochs")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=768)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--wandb-project", default="dialogue-summarizer")
    return p.parse_args()


def build_tokenize_fn(tokenizer, max_len):
    def tok(batch):
        input_ids, labels = [], []
        for dialogue, summary in zip(batch["dialogue"], batch["summary"]):
            prompt = PROMPT.format(dialogue=dialogue)
            completion = " " + summary.strip() + tokenizer.eos_token
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            ids = (p_ids + c_ids)[:max_len]
            lab = ([-100] * len(p_ids) + c_ids)[:max_len]  # mask the prompt
            input_ids.append(ids)
            labels.append(lab)
        return {"input_ids": input_ids, "labels": labels}

    return tok


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset(args.dataset)
    tok_fn = build_tokenize_fn(tokenizer, args.max_len)
    train_ds = ds["train"].map(tok_fn, batched=True, remove_columns=ds["train"].column_names)
    eval_ds = ds["validation"].map(tok_fn, batched=True, remove_columns=ds["validation"].column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", label_pad_token_id=-100)

    targs = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb",
        run_name=f"{args.model.split('/')[-1]}-samsum-lora",
    )

    import os

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}/")


if __name__ == "__main__":
    main()