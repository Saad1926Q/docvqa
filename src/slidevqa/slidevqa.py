"""Shared SlideVQA dataset and grid helpers."""

import math
from typing import Any

from datasets import load_dataset
from huggingface_hub import get_token
from PIL import Image

DATASET_ID = "NTT-hil-insight/SlideVQA"
MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"
PAGE_COLUMNS = [f"page_{index}" for index in range(1, 21)]


def load_slidevqa(split: str, *, streaming: bool):
    """Load a SlideVQA split."""
    token = get_token()
    if token is None:
        raise SystemExit("No Hugging Face token found.")

    print(f"Loading {DATASET_ID} split={split!r} ...")
    return load_dataset(DATASET_ID, split=split, streaming=streaming, token=token)


def get_page_numbers(sample: dict[str, Any], *, evidence_pages_only: bool = False) -> list[int]:
    """Return available slide numbers in deck order."""
    page_numbers = (
        sample["evidence_pages"] if evidence_pages_only else range(1, len(PAGE_COLUMNS) + 1)
    )
    return [number for number in page_numbers if sample.get(f"page_{number}") is not None]


def get_slides(sample: dict[str, Any], *, evidence_pages_only: bool = False) -> list[Image.Image]:
    """Collect slide images in deck order."""
    page_numbers = get_page_numbers(sample, evidence_pages_only=evidence_pages_only)
    slides = [sample[f"page_{number}"].convert("RGB") for number in page_numbers]
    if not slides:
        raise ValueError(f"Slide deck {sample.get('deck_name')!r} has no images")
    return slides


def concat_images(
    images: list[Image.Image], max_concat: int = 5, column_num: int = 2
) -> list[Image.Image]:
    """Build slide grids."""
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


def group_page_numbers(page_numbers: list[int], max_concat: int) -> list[list[int]]:
    """Group slide numbers exactly as concat_images groups slide images."""
    interval = max(math.ceil(len(page_numbers) / max_concat), 1)
    return [
        page_numbers[start : start + interval] for start in range(0, len(page_numbers), interval)
    ]


def grid_label(page_numbers: list[int]) -> str:
    """Describe which slides appear in a grid image."""
    if len(page_numbers) == 1:
        return f"Slide {page_numbers[0]}:"
    pages = ", ".join(str(number) for number in page_numbers)
    return f"Slides {pages}, arranged left-to-right then top-to-bottom:"
