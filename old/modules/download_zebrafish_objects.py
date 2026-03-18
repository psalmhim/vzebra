# ==============================================================
# TOOL: download_zebrafish_objects.py
# PURPOSE:
#   Download 5 types of zebrafish-relevant images from the internet:
#     1. Food (rotifers, artemia nauplii)
#     2. Plankton
#     3. Prey (paramecium images)
#     4. Neutral object (pebbles, particles)
#     5. Predator (large fish silhouettes, looming fish)
#
# OUTPUT:
#     dataset/<category>/<id>.png   (28×28 grayscale images)
#
# DEPENDENCIES:
#     pip install duckduckgo-search pillow opencv-python tqdm
# ==============================================================

import os
import io
import requests
from pathlib import Path
from tqdm import tqdm
from ddgs import DDGS

from PIL import Image
import numpy as np
import cv2


# ----------------------------------------------------------------------
# SEARCH TERMS FOR EACH CATEGORY
# ----------------------------------------------------------------------
SEARCH_QUERIES = {
    "food": [
        "rotifer microscope image",
        "artemia nauplii microscope",
        "daphnia juvenile microscope"
    ],
    "plankton": [
        "plankton microscope brightfield",
        "zooplankton micrograph",
        "phytoplankton microscope"
    ],
    "prey": [
        "paramecium microscope image",
        "euglena microscope brightfield",
    ],
    "neutral": [
        "pebbles  microscope",
        "small particles micrograph",
        "microscope debris"
    ],
    "predator": [
        "fish silhouette",
        "big fish approaching underwater",
        "looming fish view"
    ]
}


# ----------------------------------------------------------------------
# Utility: Normalize → 128×128 grayscale
# ----------------------------------------------------------------------
def preprocess_image(img_bytes, size=128):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    except Exception:
        return None

    # Resize with preserving aspect ratio
    img = img.resize((size, size), Image.BILINEAR)
    return img


# ----------------------------------------------------------------------
# Download images via DuckDuckGo
# ----------------------------------------------------------------------
def download_images_from_query(query, max_n=50):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.images(query, max_results=max_n):
            if "image" in r and r["image"].startswith("http"):
                results.append(r["image"])
    return results


# ----------------------------------------------------------------------
# Main Collector
# ----------------------------------------------------------------------
def download_dataset(base="dataset/", max_per_class=80):
    base = Path(base)
    base.mkdir(exist_ok=True)

    for category, queries in SEARCH_QUERIES.items():
        out_dir = base / category
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Downloading category: {category} ===")
        urls = []

        # gather urls from multiple search terms
        for q in queries:
            urls.extend(download_images_from_query(q, max_n=40))

        print(f"Found {len(urls)} URLs before filtering")

        saved = 0

        for i, url in enumerate(tqdm(urls)):
            if saved >= max_per_class:
                break

            # download image
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200:
                    continue
            except:
                continue

            # preprocess
            img = preprocess_image(resp.content)
            if img is None:
                continue

            # save
            img_path = out_dir / f"{saved:04d}.png"
            img.save(img_path)
            saved += 1

        print(f"Saved {saved} images for {category}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    download_dataset(base="dataset", max_per_class=80)
