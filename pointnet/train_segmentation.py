"""
Training script for PointNet Part Segmentation.

Usage:
    python -m pointnet.train_segmentation \
        --data_root ./data/shapenet \
        --num_points 2048 \
        --batch_size 16 \
        --epochs 200 \
        --lr 0.001 \
        --num_parts 50 \
        --feature_transform \
        --save_dir ./checkpoints

Reference:
    Training protocol follows the PointNet paper for part segmentation
    on the ShapeNet Part dataset.
"""

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pointnet.model import PointNetSegmentation
from pointnet.utils import PointNetSegmentationLoss, compute_mean_iou


class ShapeNetPartDataset(Dataset):
    """ShapeNet Part Segmentation Dataset.

    Expects preprocessed .npy files with the following structure:
        root/
          train/
            points/
              000000.npy   # (N, 3) point coordinates
              ...
            labels/
              000000.npy   # (N,) per-point part labels
              ...
          test/
            points/
              ...
            labels/
              ...

    Args:
        root: Root directory of the dataset.
        num_points: Number of points to sample from each shape.
        split: Dataset split ('train' or 'test').
    """

    def __init__(self, root: str, num_points: int = 2048,
                 split: str = "train"):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.split = split

        points_dir = os.path.join(root, split, "points")
        labels_dir = os.path.join(root, split, "labels")

        self.point_files = []
        self.label_files = []

        if os.path.isdir(points_dir):
            for fname in sorted(os.listdir(points_dir)):
                if fname.endswith(".npy"):
                    self.point_files.append(os.path.join(points_dir, fname))
                    self.label_files.append(
                        os.path.join(labels_dir, fname)
                    )

    def __len__(self) -> int:
        return len(self.point_files)

    def __getitem__(self, idx: int):
        points = np.load(self.point_files[idx]).astype(np.float32)
        labels = np.load(self.label_files[idx]).astype(np.int64)

        # Use only XYZ
        if points.shape[1] > 3:
            points = points[:, :3]

        # Sample fixed number of points
        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice, :]
        labels = labels[choice]

        # Normalize to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        return points, labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PointNet for Part Segmentation"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of ShapeNet Part dataset")
    parser.add_argument("--num_points", type=int, default=2048,
                        help="Number of points per sample (default: 2048)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--num_parts", type=int, default=50,
                        help="Number of part classes (default: 50)")
    parser.add_argument("--feature_transform", action="store_true",
                        help="Use feature transform T-Net")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers (default: 4)")
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    num_parts):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    for points, labels in tqdm(dataloader, desc="Training", leave=False):
        points = points.to(device)   # (B, N, 3)
        labels = labels.to(device)   # (B, N)

        optimizer.zero_grad()

        predictions, input_trans, feat_trans = model(points)
        loss = criterion(predictions, labels, feat_trans)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_mean_iou(predictions, labels, num_parts)
        num_batches += 1

    return total_loss / num_batches, total_iou / num_batches


def evaluate(model, dataloader, criterion, device, num_parts):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for points, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            points = points.to(device)
            labels = labels.to(device)

            predictions, input_trans, feat_trans = model(points)
            loss = criterion(predictions, labels, feat_trans)

            total_loss += loss.item()
            total_iou += compute_mean_iou(predictions, labels, num_parts)
            num_batches += 1

    return total_loss / num_batches, total_iou / num_batches


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets
    train_dataset = ShapeNetPartDataset(
        root=args.data_root, num_points=args.num_points, split="train"
    )
    test_dataset = ShapeNetPartDataset(
        root=args.data_root, num_points=args.num_points, split="test"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # Model
    model = PointNetSegmentation(
        num_parts=args.num_parts,
        feature_transform=args.feature_transform,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = PointNetSegmentationLoss(alpha=0.001)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.num_parts
        )
        print(f"  Train Loss: {train_loss:.4f} | Train mIoU: {train_iou:.4f}")

        test_loss, test_iou = evaluate(
            model, test_loader, criterion, device, args.num_parts
        )
        print(f"  Test  Loss: {test_loss:.4f} | Test  mIoU: {test_iou:.4f}")

        scheduler.step()

        if test_iou > best_iou:
            best_iou = test_iou
            save_path = os.path.join(args.save_dir, "best_segmentation.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
            }, save_path)
            print(f"  Saved best model (mIoU: {best_iou:.4f})")

        if epoch % 10 == 0:
            save_path = os.path.join(args.save_dir, f"seg_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iou": test_iou,
            }, save_path)

    print(f"\nTraining complete. Best test mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
