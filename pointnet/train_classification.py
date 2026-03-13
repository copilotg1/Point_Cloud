"""
Training script for PointNet 3D Classification on ModelNet40.

Usage:
    python -m pointnet.train_classification \
        --data_root ./data/modelnet40 \
        --num_points 1024 \
        --batch_size 32 \
        --epochs 200 \
        --lr 0.001 \
        --num_classes 40 \
        --feature_transform \
        --save_dir ./checkpoints

Reference:
    Training follows the protocol from the PointNet paper:
    - Adam optimizer with initial learning rate 0.001
    - Learning rate decay by 0.5 every 20 epochs
    - Batch size 32
    - 1024 points per sample
"""

import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pointnet.model import PointNetClassification
from pointnet.dataset import ModelNet40Dataset
from pointnet.utils import PointNetClassificationLoss, compute_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PointNet for 3D Object Classification"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of ModelNet40 dataset")
    parser.add_argument("--num_points", type=int, default=1024,
                        help="Number of points per sample (default: 1024)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001)")
    parser.add_argument("--num_classes", type=int, default=40,
                        help="Number of object classes (default: 40)")
    parser.add_argument("--feature_transform", action="store_true",
                        help="Use feature transform T-Net")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers (default: 4)")
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for points, labels in tqdm(dataloader, desc="Training", leave=False):
        points = points.to(device)   # (B, N, 3)
        labels = labels.to(device)   # (B,)

        optimizer.zero_grad()

        # Forward pass
        predictions, input_trans, feat_trans = model(points)

        # Compute loss
        loss = criterion(predictions, labels, feat_trans)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for points, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            points = points.to(device)
            labels = labels.to(device)

            predictions, input_trans, feat_trans = model(points)
            loss = criterion(predictions, labels, feat_trans)

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main():
    args = parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets and DataLoaders
    train_dataset = ModelNet40Dataset(
        root=args.data_root,
        num_points=args.num_points,
        split="train",
        augment=True,
    )
    test_dataset = ModelNet40Dataset(
        root=args.data_root,
        num_points=args.num_points,
        split="test",
        augment=False,
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
    model = PointNetClassification(
        num_classes=args.num_classes,
        feature_transform=args.feature_transform,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = PointNetClassificationLoss(alpha=0.001)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

        scheduler.step()

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            save_path = os.path.join(args.save_dir, "best_classification.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
            }, save_path)
            print(f"  Saved best model (accuracy: {best_accuracy:.4f})")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_path = os.path.join(args.save_dir, f"cls_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": test_acc,
            }, save_path)

    print(f"\nTraining complete. Best test accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
