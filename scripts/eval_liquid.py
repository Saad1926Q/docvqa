"""
Standalone baseline evaluation for Liquid LFM2.5-VL-1.6B on DocVQA.

Usage:
    uv run python scripts/eval_liquid.py \
        --model LiquidAI/LFM2.5-VL-1.6B \
        --split validation \
        --batch_size 8 \
        --limit 100

    # Full eval
    uv run python scripts/eval_liquid.py --batch_size 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_score(groundtruth: str, prediction: str) -> float:
    gt = " ".join(groundtruth.strip().lower().split())
    pred = " ".join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt, pred)
    length = max(len(groundtruth), len(prediction))
    if length == 0:
        return 1.0
    return 1.0 - (dist / length)


def compute_anls_metric(answers_list, predictions_list, threshold=0.5):
    per_sample = []
    for answers, pred in zip(answers_list, predictions_list, strict=True):
        scores = [anls_score(ans, pred) for ans in answers]
        best = max(scores)
        per_sample.append(best if best >= threshold else 0.0)
    return float(np.mean(per_sample)), per_sample


class LFM25VL:
    def __init__(
        self,
        model_path: str = "LiquidAI/LFM2.5-VL-1.6B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 1024,
    ):
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens

        print(f"Loading processor from {model_path} ...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        self.tokenizer = self.processor.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        print(f"Loading model from {model_path} ...")
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        print(f"Model loaded on {self.device}")

    @torch.inference_mode()
    def generate_batch(self, images: list[Image.Image], texts: list[str]) -> list[str]:
        if not images:
            return []

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            for image, text in zip(images, texts, strict=True)
        ]
        chat_inputs = [
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conversation in conversations
        ]

        inputs = self.processor(
            images=images,
            text=chat_inputs,
            padding=True,
            return_tensors="pt",
        ).to(dtype=torch.bfloat16, device=self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            num_beams=1,
            use_cache=True,
            pad_token_id=self.pad_token_id,
        )

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_length:]
        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return [text.strip() for text in decoded]


def load_docvqa(split: str = "validation", limit: int | None = None):
    print(f"Loading DocVQA {split} split from HuggingFace ...")
    dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split, trust_remote_code=True)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    samples = []
    for item in dataset:
        samples.append(
            {
                "image": item["image"],
                "question": item["question"],
                "answers": item.get("answers", [item.get("answer", "")]),
            }
        )

    print(f"Loaded {len(samples)} samples")
    return samples


def build_prompt(question: str) -> str:
    return f"{question}\nAnswer the question using a single word or phrase."


def main():
    parser = argparse.ArgumentParser(description="Evaluate LFM2.5-VL on DocVQA")
    parser.add_argument("--model", type=str, default="LiquidAI/LFM2.5-VL-1.6B")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--standard_anls", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = LFM25VL(model_path=args.model, max_new_tokens=args.max_new_tokens)
    samples = load_docvqa(split=args.split, limit=args.limit)

    print(f"\nRunning inference with batch_size={args.batch_size} ...")
    all_predictions = []
    all_answers = []

    for i in tqdm(range(0, len(samples), args.batch_size), desc="Inference"):
        batch = samples[i : i + args.batch_size]
        batch_images = [s["image"] for s in batch]
        batch_texts = [build_prompt(s["question"]) for s in batch]

        predictions = model.generate_batch(batch_images, batch_texts)
        all_predictions.extend(predictions)
        all_answers.extend([s["answers"] for s in batch])

    mean_anls, per_sample = compute_anls_metric(all_answers, all_predictions)

    print(f"\n{'=' * 50}")
    print(f"  ANLS: {mean_anls:.4f}")
    print(f"{'=' * 50}")

    results = {
        "model": args.model,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_samples": len(all_predictions),
        "anls_score": mean_anls,
    }

    results_path = output_dir / f"lfm2.5-vl_docvqa-{args.split}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    predictions_path = output_dir / f"lfm2.5-vl_docvqa-{args.split}_predictions.jsonl"
    with open(predictions_path, "w") as f:
        for i, sample in enumerate(samples):
            record = {
                "index": i,
                "question": sample["question"],
                "answers": sample["answers"],
                "prediction": all_predictions[i],
                "anls": per_sample[i],
            }
            f.write(json.dumps(record) + "\n")
    print(f"Predictions saved to {predictions_path}")

    print("\n--- Sample predictions ---")
    for i in range(min(5, len(samples))):
        print(f"\n  Q: {samples[i]['question']}")
        print(f"  GT: {samples[i]['answers']}")
        print(f"  Pred: {all_predictions[i]}")
        print(f"  ANLS: {per_sample[i]:.4f}")

    return mean_anls


if __name__ == "__main__":
    main()
