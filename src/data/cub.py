import os
import pandas as pd

ROOT = os.path.join("data", "CUB_200_2011")


def load_metadata(root=ROOT):
    images = pd.read_csv(os.path.join(root, "images.txt"), sep=" ",
                         header=None, names=["image_id", "rel_path"])
    labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ",
                         header=None, names=["image_id", "class_id"])
    split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ",
                        header=None, names=["image_id", "is_train"])
    classes = pd.read_csv(os.path.join(root, "classes.txt"), sep=" ",
                          header=None, names=["class_id", "class_folder"])

    df = (images.merge(labels, on="image_id")
                .merge(split, on="image_id")
                .merge(classes, on="class_id"))
    df["class_name"] = (df["class_folder"].str.split(".", n=1).str[1]
                        .str.replace("_", " "))
    df["label"] = df["class_id"] - 1  # 0..199
    df["path"] = df["rel_path"].apply(
        lambda p: os.path.join(root, "images", *p.split("/")))
    df["official_split"] = df["is_train"].map({1: "train", 0: "test"})
    return df