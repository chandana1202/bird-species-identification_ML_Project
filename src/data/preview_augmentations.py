import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from src.data.dataset import MEAN, STD, get_transforms, resolve_path


def main():
    os.makedirs(
        os.path.join("results", "figures"),
        exist_ok=True
    )

    df = pd.read_csv(
        os.path.join("splits", "train.csv")
    )

    row = df.sample(
        1,
        random_state=7
    ).iloc[0]

    img = Image.open(
        resolve_path(row["path"])
    ).convert("RGB")

    tf = get_transforms(
        224,
        train=True
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6)
    )

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original")

    mean = torch.tensor(MEAN)[:, None, None]
    std = torch.tensor(STD)[:, None, None]

    # Seven augmented versions
    for ax in list(axes.ravel())[1:]:
        t = tf(img) * std + mean

        ax.imshow(
            t.clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )

        ax.set_title("Augmented")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        row["class_name"]
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            "results",
            "figures",
            "augmentation_examples.png"
        ),
        dpi=150
    )

    plt.close()

    print(
        "Saved:",
        os.path.join(
            "results",
            "figures",
            "augmentation_examples.png"
        )
    )


if __name__ == "__main__":
    main()