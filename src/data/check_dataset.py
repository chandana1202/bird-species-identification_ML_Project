import os
import torch

ROOT = os.path.join("data", "CUB_200_2011")


def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA (GPU) available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    images_dir = os.path.join(ROOT, "images")

    if not os.path.isdir(images_dir):
        print("ERROR: images folder not found at", images_dir)
        return

    classes = sorted(os.listdir(images_dir))

    total = 0

    for c in classes:
        total += len(os.listdir(os.path.join(images_dir, c)))

    print("Number of classes:", len(classes))
    print("Total images:", total)
    print("First 3 classes:", classes[:3])


if __name__ == "__main__":
    main()