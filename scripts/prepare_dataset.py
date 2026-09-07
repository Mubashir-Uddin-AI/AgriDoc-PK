# -*- coding: utf-8 -*-
"""Download ONLY the exact number of images needed per class."""

from __future__ import annotations
import argparse, logging, random, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files, HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "mubashiruddin01/agridoc-pk-dataset"

CLASS_MAP = {
    "cotton_bacterial_blight": "cotton_bacterial_blight",
    "cotton_curl_virus":       "cotton_leaf_curl_virus",
    "cotton_healthy":          "cotton_healthy",
    "rice_blast":              "rice_blast",
    "rice_brown_spot":         "rice_brown_spot",
    "rice_healthy":            "rice_healthy",
    "wheat_healthy":           "wheat_healthy",
    "wheat_brown_rust":        "wheat_leaf_rust",
    "wheat_yellow_rust":       "wheat_stripe_rust",
}

OUTPUT = Path("data/processed")


def run(max_per_class, seed):
    random.seed(seed)
    api = HfApi()

    logger.info("Listing files in repo...")
    all_files = list(api.list_repo_files(REPO_ID, repo_type="dataset"))
    logger.info("Total files in repo: %d", len(all_files))

    # Group by split/class and pick only max_per_class
    to_download = []
    for split in ("train", "val"):
        for ds_class, your_class in CLASS_MAP.items():
            prefix = f"{split}/{ds_class}/"
            files = [f for f in all_files if f.startswith(prefix)]
            if not files:
                continue
            random.shuffle(files)
            if max_per_class:
                files = files[:max_per_class]
            for f in files:
                to_download.append((f, split, your_class))
            logger.info("  %s/%-25s -> %-25s  %d files selected", split, ds_class, your_class, len(files))

    logger.info("Total files to download: %d", len(to_download))

    # Download one by one
    done = 0
    failed = 0
    for repo_file, split, your_class in to_download:
        dest_dir = OUTPUT / split / your_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = Path(repo_file).name
        dest = dest_dir / fname

        if dest.exists():
            done += 1
            continue

        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_file,
                repo_type="dataset",
            )
            shutil.copy2(cached, str(dest))
            done += 1
        except Exception as e:
            failed += 1
            if failed <= 3:
                logger.warning("Failed: %s (%s)", repo_file, e)

        if (done + failed) % 50 == 0:
            logger.info("  Progress: %d/%d done, %d failed", done, len(to_download), failed)

    logger.info("Done! %d downloaded, %d failed", done, failed)
    logger.info("")
    logger.info("=== Final layout ===")
    for split in ("train", "val"):
        sd = OUTPUT / split
        if not sd.exists():
            continue
        for cls in sorted(sd.iterdir()):
            if cls.is_dir():
                n = sum(1 for f in cls.iterdir() if f.is_file())
                logger.info("  %s/%-30s  %d images", split, cls.name, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_class", type=int, default=300)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    max_pc = None if args.full else args.max_per_class
    logger.info("Mode: %s", "FULL" if max_pc is None else f"{max_pc} imgs/class")
    run(max_per_class=max_pc, seed=args.seed)


if __name__ == "__main__":
    main()
