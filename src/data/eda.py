import os
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from cub import load_metadata

FIG_DIR = os.path.join("results", "figures")
MET_DIR = os.path.join("results", "metrics")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MET_DIR, exist_ok=True)
    df = load_metadata()
    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log(f"Total images: {len(df)}")
    log(f"Number of classes: {df['label'].nunique()}")
    log(f"Official split counts: {df['official_split'].value_counts().to_dict()}")

    counts = df.groupby("class_name").size().sort_values()
    log(f"Images per class -> min {counts.iloc[0]} ({counts.index[0]}), "
        f"max {counts.iloc[-1]} ({counts.index[-1]}), "
        f"mean {counts.mean():.1f}, median {counts.median():.1f}")
    tr = df[df.official_split == "train"].groupby("label").size()
    log(f"Train images per class -> min {tr.min()}, max {tr.max()}")

    # Class distribution plot
    plt.figure(figsize=(14, 4))
    plt.bar(range(len(counts)), counts.values)
    plt.xlabel("Species (sorted by image count)")
    plt.ylabel("Number of images")
    plt.title("Images per species (CUB-200-2011)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # Image sizes, modes, corrupted files
    widths, heights, modes, bad = [], [], Counter(), []
    for p in tqdm(df["path"], desc="Checking images"):
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                widths.append(im.size[0])
                heights.append(im.size[1])
                modes[im.mode] += 1
        except Exception:
            bad.append(p)

    log(f"Corrupted/unreadable images: {len(bad)}")
    log(f"Image modes: {dict(modes)}")
    if widths:
        s = sorted(widths), sorted(heights)
        log(f"Width  -> min {s[0][0]}, median {s[0][len(s[0])//2]}, max {s[0][-1]}")
        log(f"Height -> min {s[1][0]}, median {s[1][len(s[1])//2]}, max {s[1][-1]}")

        plt.figure(figsize=(6, 5))
        plt.scatter(widths, heights, s=3, alpha=0.3)
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.title("Image dimensions")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "image_dimensions.png"), dpi=150)
        plt.close()

    # Sample grid: one image from 12 random species
    random.seed(42)
    picks = random.sample(sorted(df["label"].unique()), 12)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, lab in zip(axes.ravel(), picks):
        row = df[df.label == lab].sample(1, random_state=42).iloc[0]
        ax.imshow(Image.open(row["path"]).convert("RGB"))
        ax.set_title(row["class_name"], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sample_images.png"), dpi=150)
    plt.close()

    with open(os.path.join(MET_DIR, "eda_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("Saved figures to", FIG_DIR)


if __name__ == "__main__":
    main()