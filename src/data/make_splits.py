import os

import pandas as pd
from sklearn.model_selection import train_test_split

from cub import load_metadata


SEED = 42
VAL_FRACTION = 0.2

HASH_FILE = os.path.join(
    "results",
    "metrics",
    "image_hashes.csv"
)

OUT_DIR = "splits"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_metadata()

    log = []

    # --------------------------------------------------
    # Duplicate removal
    # --------------------------------------------------

    if os.path.exists(HASH_FILE):

        h = pd.read_csv(
            HASH_FILE
        )[
            [
                "image_id",
                "phash"
            ]
        ]

        df = df.merge(
            h,
            on="image_id"
        )

        # Find hashes present in official test set
        test_hashes = set(
            df[
                df.official_split == "test"
            ]["phash"]
        )

        # Remove training images that duplicate test images
        leak = (
            (df.official_split == "train")
            &
            df["phash"].isin(test_hashes)
        )

        log.append(
            f"Removed {int(leak.sum())} train images "
            f"duplicating a test image"
        )

        df = df[~leak]

        # Remove repeated duplicates inside training pool
        pool = df[
            df.official_split == "train"
        ]

        repeated = pool[
            pool.duplicated(
                "phash",
                keep="first"
            )
        ]

        log.append(
            f"Removed {len(repeated)} repeated duplicates "
            f"inside the train pool"
        )

        df = df[
            ~df["image_id"].isin(
                repeated["image_id"]
            )
        ]

    else:

        log.append(
            "WARNING: image_hashes.csv not found; "
            "duplicate removal skipped"
        )

    # --------------------------------------------------
    # Official test set
    # --------------------------------------------------

    test = df[
        df.official_split == "test"
    ]

    # Remaining training pool
    pool = df[
        df.official_split == "train"
    ]

    # --------------------------------------------------
    # Stratified train / validation split
    # --------------------------------------------------

    train, val = train_test_split(
        pool,
        test_size=VAL_FRACTION,
        stratify=pool["label"],
        random_state=SEED
    )

    # --------------------------------------------------
    # Leakage safety checks
    # --------------------------------------------------

    assert not (
        set(train.image_id)
        &
        set(val.image_id)
    )

    assert not (
        set(train.image_id)
        &
        set(test.image_id)
    )

    assert not (
        set(val.image_id)
        &
        set(test.image_id)
    )

    # --------------------------------------------------
    # Save split files
    # --------------------------------------------------

    cols = [
        "image_id",
        "path",
        "label",
        "class_name"
    ]

    train[cols].to_csv(
        os.path.join(
            OUT_DIR,
            "train.csv"
        ),
        index=False
    )

    val[cols].to_csv(
        os.path.join(
            OUT_DIR,
            "val.csv"
        ),
        index=False
    )

    test[cols].to_csv(
        os.path.join(
            OUT_DIR,
            "test.csv"
        ),
        index=False
    )

    (
        df[
            [
                "label",
                "class_name"
            ]
        ]
        .drop_duplicates()
        .sort_values("label")
        .to_csv(
            os.path.join(
                OUT_DIR,
                "classes.csv"
            ),
            index=False
        )
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    log.append(
        f"Train: {len(train)}  "
        f"Val: {len(val)}  "
        f"Test: {len(test)}"
    )

    log.append(
        f"Classes in train: {train.label.nunique()}, "
        f"val: {val.label.nunique()}, "
        f"test: {test.label.nunique()}"
    )

    print()
    print("\n".join(log))

    with open(
        os.path.join(
            "results",
            "metrics",
            "split_summary.txt"
        ),
        "w"
    ) as f:

        f.write(
            "\n".join(log)
        )


if __name__ == "__main__":
    main()