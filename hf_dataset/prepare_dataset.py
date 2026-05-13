#!/usr/bin/env python3
"""
Download the source classification finetuning dataset (config: all only),
rename `label` -> `main_category`, save locally and/or push to a new Hub repo.

Requires: pip install -r requirements.txt
Push auth: set HF_TOKEN (write token) or run `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO
from pathlib import Path

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi


SOURCE_DATASET = "OpenWhistleNeurIPS26/OpenWhistle-Classification-Finetuning"
SOURCE_CONFIG = "all"
OLD_LABEL_COL = "label"
NEW_LABEL_COL = "main_category"

# Published dataset on the Hugging Face Hub (see hf_dataset/README.md).
TARGET_DATASET_REPO = "dolphinteam/Whistle-Classification"


def load_transformed() -> DatasetDict:
    raw: DatasetDict = load_dataset(SOURCE_DATASET, SOURCE_CONFIG)
    if OLD_LABEL_COL not in raw["train"].column_names:
        raise KeyError(
            f"Expected column {OLD_LABEL_COL!r} in train split; "
            f"got {raw['train'].column_names}"
        )
    return raw.rename_column(OLD_LABEL_COL, NEW_LABEL_COL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-to-disk",
        type=Path,
        default=None,
        help="Optional directory to write DatasetDict (Arrow).",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help=f"Target dataset repo id (expected: {TARGET_DATASET_REPO}).",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path(__file__).resolve().parent / "README.md",
        help="README.md to upload with the dataset (dataset card).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repo when pushing (if repo does not exist).",
    )
    args = parser.parse_args()

    if args.save_to_disk is None and args.push_to_hub is None:
        parser.error("Provide --save-to-disk and/or --push-to-hub")

    ds = load_transformed()

    if args.save_to_disk is not None:
        args.save_to_disk.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(args.save_to_disk))
        print(f"Saved DatasetDict to {args.save_to_disk}")

    if args.push_to_hub is not None:
        token = os.environ.get("HF_TOKEN")
        ds.push_to_hub(args.push_to_hub, private=args.private, token=token)
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.push_to_hub}")

        readme_path = args.readme
        if readme_path.is_file():
            api = HfApi(token=token)
            card = readme_path.read_text(encoding="utf-8")
            api.upload_file(
                path_or_fileobj=BytesIO(card.encode("utf-8")),
                path_in_repo="README.md",
                repo_id=args.push_to_hub,
                repo_type="dataset",
                token=token,
            )
            print(f"Uploaded dataset card from {readme_path}")
        else:
            print(f"No README at {readme_path}; skipped card upload")


if __name__ == "__main__":
    main()
