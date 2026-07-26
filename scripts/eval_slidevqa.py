"""Evaluate LiquidAI/LFM2.5-VL-1.6B on SlideVQA.

Evaluation approach adapted from VLMEvalKit, using the Hugging Face dataset.

Usage:
    uv run python scripts/eval_slidevqa.py --limit 20
    uv run python scripts/eval_slidevqa.py --batch_size 2
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import get_token
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

DATASET_ID = "NTT-hil-insight/SlideVQA"
MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"
PAGE_COLUMNS = [f"page_{index}" for index in range(1, 21)]


def load_slidevqa(split: str):
    """Load a SlideVQA split."""
    token = get_token()
    if token is None:
        raise SystemExit("No Hugging Face token found.")

    print(f"Loading {DATASET_ID} split={split!r} ...")
    return load_dataset(DATASET_ID, split=split, streaming=True, token=token)


def get_slides(sample: dict[str, Any]) -> list[Image.Image]:
    """Collect non-empty page_1 ... page_20 images in deck order."""
    slides = [
        sample[column].convert("RGB") for column in PAGE_COLUMNS if sample[column] is not None
    ]
    if not slides:
        raise ValueError(f"Slide deck {sample.get('deck_name')!r} has no images")
    return slides


def concat_images(
    images: list[Image.Image], max_concat: int = 5, column_num: int = 2
) -> list[Image.Image]:
    """Build slide grids.

    max_concat is the maximum number of output grid images, not the number of
    slides in each grid. For 20 slides and max_concat=5, this creates five grids
    containing four slides each.
    """
    interval = max(math.ceil(len(images) / max_concat), 1)
    concatenated_images = []

    for start in range(0, len(images), interval):
        batch = images[start : start + interval]
        width = batch[0].width * column_num
        rows = math.ceil(len(batch) / column_num)
        height = batch[0].height * rows
        grid = Image.new("RGB", (width, height), "white")

        for index, image in enumerate(batch):
            x = (index % column_num) * image.width
            y = (index // column_num) * image.height
            grid.paste(image, (x, y))

        concatenated_images.append(grid)

    return concatenated_images


def build_prompt(question: str) -> str:
    """Request a directly scorable answer instead of using a GPT extraction judge."""
    return f"{question}\nAnswer using only the short answer, without explanation."


def levenshtein_distance(source: str, target: str) -> int:
    """Compute character-level Levenshtein distance."""
    if len(source) > len(target):
        source, target = target, source

    distances: list[int] | range = range(len(source) + 1)
    for target_index, target_char in enumerate(target):
        next_distances = [target_index + 1]
        for source_index, source_char in enumerate(source):
            if source_char == target_char:
                next_distances.append(distances[source_index])
            else:
                next_distances.append(
                    1
                    + min(
                        distances[source_index],
                        distances[source_index + 1],
                        next_distances[-1],
                    )
                )
        distances = next_distances
    return distances[-1]


def anls_score(answer: str, prediction: str, threshold: float = 0.5) -> float:
    """Compute thresholded ANLS."""
    length = max(len(answer), len(prediction))
    if length == 0:
        return 0.0

    score = 1.0 - (levenshtein_distance(answer, prediction) / length)
    return score if score > threshold else 0.0


def word_f1(answer: str, prediction: str) -> float:
    """Compute whitespace-token overlap F1."""
    answer_words = answer.strip().split()
    prediction_words = prediction.strip().split()
    if not answer_words or not prediction_words:
        return 0.0

    recall = sum(word in answer_words for word in prediction_words) / len(answer_words)
    precision = sum(word in answer_words for word in prediction_words) / len(prediction_words)
    if recall + precision <= 1e-4:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def normalize_answer(answer: Any) -> str:
    """Normalize a ground-truth answer."""
    if answer is None:
        return "not answerable"
    return re.sub("\n", "", str(answer)).lower()


def score_prediction(answer: Any, prediction: str) -> dict[str, float]:
    """Compute SlideVQA ANLS, exact match, and word F1."""
    normalized_answer = normalize_answer(answer)
    normalized_prediction = str(prediction).lower()
    return {
        "anls": anls_score(normalized_answer, normalized_prediction),
        "em": float(normalized_answer.strip() == normalized_prediction.strip()),
        "f1": word_f1(normalized_answer, normalized_prediction),
    }


class LFM25VL:
    """Minimal batched multi-image inference wrapper for LFM2.5-VL."""

    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        self.device = torch.device("cuda")
        self.max_new_tokens = max_new_tokens

        print(f"Loading processor from {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        self.pad_token_id = (
            self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        )

        print(f"Loading model from {model_id} ...")
        model: Any = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self.model = model.to(self.device).eval()
        print(f"Model loaded on {self.device}")

    @torch.inference_mode()
    def generate_batch(
        self, image_batches: list[list[Image.Image]], questions: list[str]
    ) -> list[str]:
        """Generate one answer per deck from all of its concatenated grid images."""
        conversations = []
        for images, question in zip(image_batches, questions, strict=True):
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": build_prompt(question)})
            conversations.append([{"role": "user", "content": content}])

        prompts = [
            self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            for conversation in conversations
        ]
        inputs = self.processor(
            images=image_batches,
            text=prompts,
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
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [response.strip() for response in responses]


def batched(iterable, batch_size: int, limit: int | None):
    """Yield lists from an iterable dataset without materializing the full split."""
    batch = []
    for seen, sample in enumerate(iterable):
        if limit is not None and seen >= limit:
            break
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run SlideVQA inference and save aggregate and per-sample results."""
    dataset = load_slidevqa(args.split)
    model = LFM25VL(args.model, args.max_new_tokens)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"lfm2.5-vl_slidevqa-{args.split}_predictions.jsonl"

    totals = {"anls": 0.0, "em": 0.0, "f1": 0.0}
    count = 0
    split_size = dataset.info.splits[args.split].num_examples
    progress_total = split_size if args.limit is None else min(args.limit, split_size)
    progress = tqdm(total=progress_total, desc="SlideVQA")

    with predictions_path.open("w") as predictions_file:
        for samples in batched(dataset, args.batch_size, args.limit):
            image_batches = [
                concat_images(
                    get_slides(sample),
                    max_concat=args.max_concat,
                    column_num=args.column_num,
                )
                for sample in samples
            ]
            questions = [sample["question"] for sample in samples]
            predictions = model.generate_batch(image_batches, questions)

            for sample, prediction, grids in zip(samples, predictions, image_batches, strict=True):
                scores = score_prediction(sample["answer"], prediction)
                for metric in totals:
                    totals[metric] += scores[metric]
                count += 1

                record = {
                    "qa_id": sample["qa_id"],
                    "deck_name": sample["deck_name"],
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "prediction": prediction,
                    "arithmetic_expression": sample["arithmetic_expression"],
                    "evidence_pages": sample["evidence_pages"],
                    "num_grids": len(grids),
                    **scores,
                }
                predictions_file.write(json.dumps(record) + "\n")
                predictions_file.flush()

            progress.update(len(samples))

    progress.close()
    if count == 0:
        raise RuntimeError("No SlideVQA samples were evaluated")

    results = {
        "model": args.model,
        "dataset": DATASET_ID,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_samples": count,
        "max_concat": args.max_concat,
        "column_num": args.column_num,
        **{metric: total / count for metric, total in totals.items()},
    }
    results_path = output_dir / f"lfm2.5-vl_slidevqa-{args.split}_results.json"
    with results_path.open("w") as results_file:
        json.dump(results, results_file, indent=2)

    print(f"\nANLS: {results['anls']:.4f}")
    print(f"EM:   {results['em']:.4f}")
    print(f"F1:   {results['f1']:.4f}")
    print(f"Results: {results_path}")
    print(f"Predictions: {predictions_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_concat", type=int, default=5)
    parser.add_argument("--column_num", type=int, default=2)
    parser.add_argument("--output_dir", default="results")
    return parser.parse_args()


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
