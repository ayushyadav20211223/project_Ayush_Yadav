# train_and_test.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

# pull in your existing config, dataset and model
from config import (
    batchsize,
    epochs,
    learning_rate,
    data_path,
    loss_fn,
)
from dataset import TrafficSignDataset
from model import MyCustomModel

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    # 1. device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 2. dataset + train/test split
    full_dataset = TrafficSignDataset(data_dir=data_path)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_ds, test_ds = random_split(full_dataset, [n_train, n_test])
    print(f"🔢 Totals – train: {n_train}, test: {n_test}")

    train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batchsize, shuffle=False)

    # 3. model / optimizer / loss
    model     = MyCustomModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = loss_fn   # nn.CrossEntropyLoss()

    # 4. training loop with best‐model checkpointing
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss,  test_acc  = eval_one_epoch(model, test_loader,  criterion, device)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  ▶ Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}%")
        print(f"  🧪  Test Loss: {test_loss:.4f} |  Test Acc: {test_acc*100:5.2f}%")

        # save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"  💾 New best model saved (acc {best_acc*100:5.2f}%)")

    # 5. final save
    torch.save(model.state_dict(), "checkpoints/final_model.pth")
    print("\n✅ Training complete. Final model in checkpoints/final_model.pth")

if __name__ == "__main__":
    main()
