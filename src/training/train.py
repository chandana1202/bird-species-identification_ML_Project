import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import BirdDataset, get_transforms
from src.models.factory import build_model


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    max_batches=None
):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    correct = 0
    n = 0

    with torch.set_grad_enabled(training):

        for i, (x, y) in enumerate(loader):

            if max_batches and i >= max_batches:
                break

            x = x.to(device)
            y = y.to(device)

            out = model(x)

            loss = criterion(out, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)

            correct += (
                out.argmax(1) == y
            ).sum().item()

            n += x.size(0)

    return total_loss / n, correct / n


def set_backbone_trainable(model, head, flag):

    head_ids = {
        id(p)
        for p in head.parameters()
    }

    for p in model.parameters():

        if id(p) not in head_ids:
            p.requires_grad = flag


def plot_history(hist, tag):

    fig_dir = os.path.join(
        "results",
        "figures"
    )

    # Accuracy plot
    plt.figure(figsize=(6, 4))

    plt.plot(
        hist["epoch"],
        hist["train_acc"],
        label="Train"
    )

    plt.plot(
        hist["epoch"],
        hist["val_acc"],
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.title(
        f"Accuracy ({tag})"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            fig_dir,
            f"{tag}_accuracy.png"
        ),
        dpi=150
    )

    plt.close()

    # Loss plot
    plt.figure(figsize=(6, 4))

    plt.plot(
        hist["epoch"],
        hist["train_loss"],
        label="Train"
    )

    plt.plot(
        hist["epoch"],
        hist["val_loss"],
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.title(
        f"Loss ({tag})"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            fig_dir,
            f"{tag}_loss.png"
        ),
        dpi=150
    )

    plt.close()


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="baseline",
        choices=[
            "baseline",
            "resnet50",
            "efficientnet_b0"
        ]
    )

    ap.add_argument(
        "--img_size",
        type=int,
        default=224
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    ap.add_argument(
        "--freeze_epochs",
        type=int,
        default=3
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=5
    )

    ap.add_argument(
        "--num_workers",
        type=int,
        default=2
    )

    ap.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Smoke test only"
    )

    ap.add_argument(
        "--tag",
        default=None
    )

    args = ap.parse_args()

    tag = args.tag or args.model

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            "results",
            "figures"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            "results",
            "metrics"
        ),
        exist_ok=True
    )

    # -------------------------
    # Datasets
    # -------------------------

    train_ds = BirdDataset(
        os.path.join(
            "splits",
            "train.csv"
        ),
        get_transforms(
            args.img_size,
            train=True
        )
    )

    val_ds = BirdDataset(
        os.path.join(
            "splits",
            "val.csv"
        ),
        get_transforms(
            args.img_size,
            train=False
        )
    )

    # -------------------------
    # DataLoaders
    # -------------------------

    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin
    )

    # -------------------------
    # Model
    # -------------------------

    model, head = build_model(
        args.model,
        num_classes=200,
        pretrained=True
    )

    model.to(device)

    # -------------------------
    # Freeze configuration
    # -------------------------

    freeze_epochs = (
        0
        if args.model == "baseline"
        else args.freeze_epochs
    )

    head_ids = {
        id(p)
        for p in head.parameters()
    }

    head_params = [
        p
        for p in model.parameters()
        if id(p) in head_ids
    ]

    back_params = [
        p
        for p in model.parameters()
        if id(p) not in head_ids
    ]

    groups = [
        {
            "params": head_params,
            "lr": args.lr
        }
    ]

    if back_params:

        groups.append(
            {
                "params": back_params,
                "lr": args.lr * 0.1
            }
        )

    # -------------------------
    # Optimizer / Scheduler
    # -------------------------

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Training state
    # -------------------------

    best_acc = 0.0
    bad_epochs = 0
    rows = []

    ckpt_path = os.path.join(
        "models",
        f"{tag}_best.pth"
    )

    # -------------------------
    # Training loop
    # -------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        if back_params:

            set_backbone_trainable(
                model,
                head,
                epoch > freeze_epochs
            )

        t0 = time.time()

        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            args.max_batches
        )

        va_loss, va_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            None,
            args.max_batches
        )

        scheduler.step()

        rows.append(
            dict(
                epoch=epoch,
                train_loss=tr_loss,
                train_acc=tr_acc,
                val_loss=va_loss,
                val_acc=va_acc,
                lr=optimizer.param_groups[0]["lr"]
            )
        )

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {tr_loss:.3f} "
            f"acc {tr_acc:.3f} | "
            f"val loss {va_loss:.3f} "
            f"acc {va_acc:.3f} | "
            f"{elapsed:.0f}s"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if va_acc > best_acc:

            best_acc = va_acc
            bad_epochs = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model,
                    "num_classes": 200,
                    "img_size": args.img_size,
                    "epoch": epoch,
                    "val_acc": va_acc
                },
                ckpt_path
            )

        else:

            bad_epochs += 1

            if bad_epochs >= args.patience:

                print(
                    "Early stopping."
                )

                break

    # -------------------------
    # Save history
    # -------------------------

    hist = pd.DataFrame(rows)

    hist.to_csv(
        os.path.join(
            "results",
            "metrics",
            f"{tag}_history.csv"
        ),
        index=False
    )

    plot_history(
        hist,
        tag
    )

    print(
        f"Best validation accuracy: "
        f"{best_acc:.4f} "
        f"(saved to {ckpt_path})"
    )


if __name__ == "__main__":
    main()