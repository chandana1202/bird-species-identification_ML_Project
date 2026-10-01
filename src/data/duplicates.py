import os

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cub import load_metadata


MET_DIR = os.path.join("results", "metrics")


def main():
    os.makedirs(MET_DIR, exist_ok=True)

    df = load_metadata()

    hashes = []

    for p in tqdm(
        df["path"],
        desc="Hashing images"
    ):
        with Image.open(p) as im:
            hashes.append(
                str(
                    imagehash.phash(
                        im.convert("RGB")
                    )
                )
            )

    df["phash"] = hashes

    # Save hash for every image
    df[
        [
            "image_id",
            "rel_path",
            "label",
            "official_split",
            "phash"
        ]
    ].to_csv(
        os.path.join(
            MET_DIR,
            "image_hashes.csv"
        ),
        index=False
    )

    # Find images sharing the same hash
    dup = df[
        df.duplicated(
            "phash",
            keep=False
        )
    ].sort_values("phash")

    groups = list(
        dup.groupby("phash")
    )

    # Duplicate groups crossing train/test
    cross_split = [
        g
        for _, g in groups
        if g["official_split"].nunique() > 1
    ]

    # Duplicate groups belonging to different species
    cross_class = [
        g
        for _, g in groups
        if g["label"].nunique() > 1
    ]

    print(
        f"Images sharing a hash with another image: "
        f"{len(dup)}"
    )

    print(
        f"Duplicate groups: {len(groups)}"
    )

    print(
        f"Groups spanning train AND test: "
        f"{len(cross_split)}"
    )

    print(
        f"Groups spanning different species: "
        f"{len(cross_class)}"
    )

    # Save duplicate information
    dup[
        [
            "image_id",
            "rel_path",
            "official_split",
            "phash"
        ]
    ].to_csv(
        os.path.join(
            MET_DIR,
            "duplicate_groups.csv"
        ),
        index=False
    )


if __name__ == "__main__":
    main()