# -*- coding: utf-8 -*-
"""Fast dataset downloader for agridoc-pk-dataset.

Downloads images directly from HuggingFace by URL -- no streaming needed.
Runs 10 parallel threads for fast download.

Usage:
    python scripts/fast_download.py                     # 300 imgs/class
    python scripts/fast_download.py --max_per_class 100 # 100 imgs/class (ultra fast)
    python scripts/fast_download.py --full              # all images
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# HuggingFace base URL for direct file download
HF_BASE = "https://huggingface.co/datasets/mubashiruddin01/agridoc-pk-dataset/resolve/main"

# Mapping: local class name  ->  folder name in the HF repo
# (discovered from HF API: train/cotton_curl_virus/... not cotton_leaf_curl_virus)
CLASS_MAP = {
    "cotton_bacterial_blight": "cotton_bacterial_blight",
    "cotton_healthy":          "cotton_healthy",
    "cotton_leaf_curl_virus":  "cotton_curl_virus",   # repo folder differs
    "rice_blast":              "rice_blast",
    "rice_brown_spot":         "rice_brown_spot",
    "rice_healthy":            "rice_healthy",
    "wheat_healthy":           "wheat_healthy",
    "wheat_leaf_rust":         "wheat_leaf_rust",
    "wheat_loose_smut":        "wheat_loose_smut",
    "wheat_stripe_rust":       "wheat_stripe_rust",
}

# train/val/test split ratios applied AFTER download
SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}

OUTPUT_ROOT = Path("data/processed")
MAX_RETRIES = 3
TIMEOUT = 30


def download_file(url: str, dest: Path, retries: int = MAX_RETRIES) -> bool:
    """Download a single file with retry logic. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True  # already downloaded

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "agridoc-downloader/1.0"})
            with urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return True
        except (URLError, HTTPError, OSError) as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                logger.warning("FAIL [%d tries] %s -> %s", retries, url, e)
    return False


def download_class(
    local_class: str,
    repo_folder: str,
    max_per_class: int | None,
    workers: int,
) -> list[Path]:
    """Download images for one class. Returns list of downloaded file paths."""

    # Build URL list: images are named <folder>_0.jpg, <folder>_1.jpg, ...
    # We probe with a HEAD-like approach: try indices 0..N until 404
    # For speed, we download the first max_per_class indices (0, 1, 2, ...)
    # The repo has sequential naming so this is deterministic.

    tmp_dir = OUTPUT_ROOT / "_tmp" / local_class
    tmp_dir.mkdir(parents=True, exist_ok=True)

    limit = max_per_class if max_per_class else 99999
    urls = []
    for i in range(limit):
        filename = f"{repo_folder}_{i}.jpg"
        url = f"{HF_BASE}/train/{repo_folder}/{filename}"
        dest = tmp_dir / filename
        urls.append((url, dest))

    downloaded: list[Path] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, url, dest): dest for url, dest in urls}
        for future in as_completed(futures):
            dest = futures[future]
            try:
                ok = future.result()
                if ok and dest.exists():
                    downloaded.append(dest)
                elif not ok:
                    failed += 1
                    # Stop if we start getting 404s (beyond dataset size)
                    if failed > 5 and len(downloaded) > 0:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
            except Exception as e:
                failed += 1

    logger.info("  %s: downloaded %d images (%d failed)", local_class, len(downloaded), failed)
    return downloaded


def organize_splits(local_class: str, files: list[Path]):
    """Move files from _tmp into train/val/test splits."""
    random.shuffle(files)
    n = len(files)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:],
    }

    total_moved = 0
    for split_name, split_files in splits.items():
        dest_dir = OUTPUT_ROOT / split_name / local_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        for j, src in enumerate(split_files):
            dst = dest_dir / f"{local_class}_{j:05d}.jpg"
            shutil.move(str(src), str(dst))
            total_moved += 1

    return total_moved


def main():
    parser = argparse.ArgumentParser(description="Fast agridoc-pk-dataset downloader")
    parser.add_argument("--max_per_class", type=int, default=300,
                        help="Max images per class (default 300). Use --full for all.")
    parser.add_argument("--full", action="store_true",
                        help="Download all available images.")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel download threads per class (default 10).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    max_per_class = None if args.full else args.max_per_class

    logger.info("=" * 60)
    logger.info("AgriDoc-PK Fast Downloader")
    logger.info("Mode   : %s", "FULL" if max_per_class is None else f"{max_per_class} images/class")
    logger.info("Workers: %d parallel threads", args.workers)
    logger.info("Output : %s", OUTPUT_ROOT.resolve())
    logger.info("=" * 60)

    grand_total = 0
    t_start = time.time()

    for local_class, repo_folder in CLASS_MAP.items():
        logger.info("Downloading class: %s ...", local_class)
        t0 = time.time()
        files = download_class(local_class, repo_folder, max_per_class, args.workers)
        moved = organize_splits(local_class, files)
        grand_total += moved
        logger.info("  Done in %.1fs", time.time() - t0)

    # Cleanup temp folder
    tmp_root = OUTPUT_ROOT / "_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Total images saved : %d", grand_total)
    logger.info("Total time         : %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("Output             : %s", OUTPUT_ROOT.resolve())
    logger.info("")
    logger.info("Next step:")
    logger.info("  python models/train.py --data_dir data/processed --epochs 10 --batch_size 16")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
