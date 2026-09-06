# -*- coding: utf-8 -*-
"""Fast dataset downloader -- downloads PNG files directly by class folder.
Much faster than streaming because it fetches only the folders needed.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --max_per_class 300
    python scripts/prepare_dataset.py --full
"""

from __future__ import annotations
import argparse, logging, random, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ID = "mubashiruddin01/agridoc-pk-dataset"

# Map: dataset_folder_class -> your CLASS_NAMES (from core/classifier.py)
CLASS_MAP = {
    "cotton_bacterial_blight": "cotton_bacterial_blight",
    "cotton_curl_virus":        "cotton_leaf_curl_virus",
    "cotton_healthy":           "cotton_healthy",
    "rice_blast":               "rice_blast",
    "rice_brown_spot":          "rice_brown_spot",
    "rice_healthy":             "rice_healthy",
    "wheat_healthy":            "wheat_healthy",
    "wheat_brown_rust":         "wheat_leaf_rust",
    "wheat_yellow_rust":        "wheat_stripe_rust",
    # wheat_loose_smut has no match in dataset -- skipped
    # rice_bacterial_blight is extra -- skipped
}

OUTPUT_ROOT = Path("data/processed")


def download_file(repo_file: str, dest: Path, token):
    """Download a single file from the HF dataset repo."""
    try:
        src = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_file,
            repo_type="dataset",
            token=token,
            local_dir=str(dest.parent.parent.parent / ".hf_cache"),
        )
        shutil.copy2(src, dest)
        return True
    except Exception as e:
        logger.debug("Failed %s: %s", repo_file, e)
        return False


def run(max_per_class, token, seed, workers):
    random.seed(seed)

    logger.info("Fetching file list from HuggingFace...")
    all_files = list(list_repo_files(REPO_ID, repo_type="dataset", token=token))
    logger.info("Total files in repo: %d", len(all_files))

    # Group files by (split, dataset_class)
    file_groups: dict[tuple, list[str]] = {}
    for f in all_files:
        parts = f.split("/")
        if len(parts) < 3:
            continue
        split, ds_class = parts[0], parts[1]
        if ds_class not in CLASS_MAP:
            continue
        if split not in ("train", "val"):
            continue
        key = (split, ds_class)
        file_groups.setdefault(key, []).append(f)

    logger.info("Classes found: %s", sorted(set(k[1] for k in file_groups)))

    # Build download queue with limit
    download_tasks: list[tuple[str, Path]] = []
    for (split, ds_class), files in file_groups.items():
        your_class = CLASS_MAP[ds_class]
        random.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        dest_dir = OUTPUT_ROOT / split / your_class
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            fname = Path(f).name
            dest = dest_dir / fname
            if dest.exists():
                continue  # skip already downloaded
            download_tasks.append((f, dest))

    logger.info("Files to download: %d  (workers=%d)", len(download_tasks), workers)

    if not download_tasks:
        logger.info("All files already present! Nothing to download.")
        return

    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_file, repo_f, dest, token): (repo_f, dest)
            for repo_f, dest in download_tasks
        }
        for future in as_completed(futures):
            ok = future.result()
            if ok:
                done += 1
            else:
                failed += 1
            if (done + failed) % 50 == 0:
                logger.info("  Downloaded %d / %d (failed: %d)", done, len(download_tasks), failed)

    logger.info("Done! Downloaded %d files. Failed: %d", done, failed)

    # Summary
    logger.info("\n=== Final dataset layout ===")
    for split in ("train", "val"):
        split_dir = OUTPUT_ROOT / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            count = sum(1 for _ in cls_dir.iterdir())
            logger.info("  %s/%-30s  %d images", split, cls_dir.name, count)

    logger.info("\nRun training: python models/train.py --data_dir data/processed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_class", type=int, default=300,
                        help="Max images per class per split (default 300)")
    parser.add_argument("--full", action="store_true",
                        help="Download all images")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download threads (default 8)")
    args = parser.parse_args()

    max_pc = None if args.full else args.max_per_class
    logger.info("Mode: %s", "FULL" if max_pc is None else f"{max_pc} imgs/class")
    run(max_per_class=max_pc, token=args.token, seed=args.seed, workers=args.workers)


if __name__ == "__main__":
    main()
