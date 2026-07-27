"""LoRA SFT for LFM2.5-VL on SlideVQA with Unsloth.

Trains one adapter on two tasks:
  1. all numbered slide grids + question -> evidence page numbers
  2. gold evidence slide grids + question -> short answer

Usage:
    uv run python scripts/train_slidevqa.py --preview --limit 2
    uv run python scripts/train_slidevqa.py --limit 16 --max_steps 2
    uv run python scripts/train_slidevqa.py --num_train_epochs 1
"""

# ruff: noqa: I001

from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator

import argparse
from itertools import islice
from typing import Any, cast

from datasets import load_dataset
from huggingface_hub import get_token
from PIL import Image

from docvqa.slidevqa import (
    DATASET_ID,
    MODEL_ID,
    concat_images,
    get_page_numbers,
    get_slides,
    grid_label,
    group_page_numbers,
)
from trl import SFTConfig, SFTTrainer


def selection_instruction(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Which slide(s) contain the evidence needed to answer the question? "
        "Return only slide numbers, separated by commas."
    )


def answer_instruction(question: str) -> str:
    return f"Question: {question}\nAnswer using only the short answer, without explanation."


def evidence_target(sample: dict[str, Any]) -> str:
    return ", ".join(str(page) for page in sample["evidence_pages"])


def slide_content(
    images: list[Image.Image], page_groups: list[list[int]], instruction: str
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image, page_numbers in zip(images, page_groups, strict=True):
        content.extend(
            [
                {"type": "text", "text": grid_label(page_numbers)},
                {"type": "image", "image": image},
            ]
        )
    content.append({"type": "text", "text": instruction})
    return content


class SlideVQASFTDataset:
    def __init__(self, dataset, *, max_concat: int, column_num: int, task: str) -> None:
        self.dataset = dataset
        self.max_concat = max_concat
        self.column_num = column_num
        self.tasks = ["selection", "answer"] if task == "both" else [task]

    def __len__(self) -> int:
        return len(self.dataset) * len(self.tasks)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index // len(self.tasks)]
        task = self.tasks[index % len(self.tasks)]
        return self.selection_record(sample) if task == "selection" else self.answer_record(sample)

    def selection_record(self, sample: dict[str, Any]) -> dict[str, Any]:
        images = concat_images(
            get_slides(sample),
            max_concat=self.max_concat,
            column_num=self.column_num,
        )
        page_groups = group_page_numbers(get_page_numbers(sample), self.max_concat)
        return self.record(
            slide_content(images, page_groups, selection_instruction(sample["question"])),
            evidence_target(sample),
        )

    def answer_record(self, sample: dict[str, Any]) -> dict[str, Any]:
        images = concat_images(
            get_slides(sample, evidence_pages_only=True),
            max_concat=self.max_concat,
            column_num=self.column_num,
        )
        page_groups = group_page_numbers(
            get_page_numbers(sample, evidence_pages_only=True), self.max_concat
        )
        return self.record(
            slide_content(images, page_groups, answer_instruction(sample["question"])),
            str(sample["answer"]),
        )

    @staticmethod
    def record(user_content: list[dict[str, Any]], target: str) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ]
        }


def load_train_split(limit: int | None) -> Any:
    token = get_token()
    if token is None:
        raise SystemExit("No Hugging Face token found.")

    if limit is not None:
        dataset = load_dataset(DATASET_ID, split="train", streaming=True, token=token)
        return list(islice(dataset, limit))
    return load_dataset(DATASET_ID, split="train", token=token)


def build_dataset(args: argparse.Namespace) -> SlideVQASFTDataset:
    raw_dataset = load_train_split(args.limit)
    dataset = SlideVQASFTDataset(
        raw_dataset,
        max_concat=args.max_concat,
        column_num=args.column_num,
        task=args.task,
    )
    print(f"Loaded {len(raw_dataset)} SlideVQA train samples -> {len(dataset)} SFT records")
    return dataset


def train(args: argparse.Namespace) -> None:
    dataset = build_dataset(args)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
    )

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=cast(Any, dataset),
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=args.seed,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            report_to="wandb" if args.wandb else "none",
            run_name=args.run_name,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_seq_length,
        ),
    )
    trainer.train()
    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print(f"Saved LoRA adapter to {args.adapter_dir}")


def preview(args: argparse.Namespace) -> None:
    dataset = build_dataset(args)
    for index in range(min(2, len(dataset))):
        print(dataset[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--output_dir", default="outputs/slidevqa_sft")
    parser.add_argument("--adapter_dir", default="checkpoints/lfm2.5-vl-slidevqa-lora")
    parser.add_argument("--task", choices=["both", "selection", "answer"], default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--max_concat", type=int, default=5)
    parser.add_argument("--column_num", type=int, default=2)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", default="lfm2.5-vl-slidevqa-sft")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview:
        preview(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
