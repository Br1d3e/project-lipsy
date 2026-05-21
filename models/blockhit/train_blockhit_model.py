import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from models.window_dataset import WindowDataset
from models.blockhit.blockhit_datasets import BlockhitDataset

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt


SEQ_LEN = 15
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
CLASS_WEIGHTS = torch.tensor([1.0, 8.0], dtype=torch.float32)    # baseline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class BlockhitStackedMLP(nn.Module):
    def __init__(self, input_dim=14 * SEQ_LEN, output_dim=2, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or [512, 256, 128, 64]
        layers = []
        in_features = input_dim
        for out_features in hidden_channels:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            in_features = out_features
        layers.append(nn.Linear(in_features, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.layers(x)

class Blockhit1dCNN(nn.Module):
    def __init__(self, input_dim=14, output_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.layers(x)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_model_checkpoint(model, save_path, checkpoint_meta=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    if checkpoint_meta:
        checkpoint.update(checkpoint_meta)
    torch.save(checkpoint, save_path)


def get_X_Y(dataset):
    X = []
    Y = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X.append(x)
        Y.append(y)
    return torch.stack(X), torch.stack(Y)

def print_class_distribution(dataset, name):
    ys = get_X_Y(dataset)[1]
    cls = blockhit_to_class(ys).numpy()
    unique, counts = np.unique(cls, return_counts=True)
    print(f"\n{name} class distribution:")
    for u, c in zip(unique, counts):
        print(f"  class {u} / blockhit {u}: {c} ({c / len(cls):.2%})")

def blockhit_to_class(y):
    y = y.reshape(-1)
    return torch.clamp(torch.round(y).long(), 0, 1)

def train(model, dataloader, val_loader, save_path, checkpoint_meta=None,
          epochs=EPOCHS, lr=LEARNING_RATE, patience=25):
    device = get_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10)
    class_weights = CLASS_WEIGHTS.to(device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_loss = 1e9
    best_epoch = -1
    best_state_dict = None
    loss_count = 0
    epochs_run = 0
    for epoch in range(epochs):
        epochs_run = epoch + 1
        total_loss = 0.0
        model.train()

        for xb, yb in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            yb = blockhit_to_class(yb)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        val_loss = eval_model(model, val_loader)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val PPL: {np.exp(val_loss):.6f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_model_checkpoint(model, save_path, checkpoint_meta)
            loss_count = 0
            print("model saved")
        else:
            loss_count += 1

        if loss_count > patience:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return {
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
    }


def eval_model(model, dataloader, graph=False, diagnostics=False):
    device = get_device()
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_count = 0
    class_weights = CLASS_WEIGHTS.to(device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    Y_true = []
    Y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_class = blockhit_to_class(y)
            pred = model(x)
            Y_true.append(y_class.cpu().numpy())
            Y_pred.append(pred.cpu().numpy())

            total_loss += criterion(pred, y_class).item() * y_class.numel()
            total_count += y_class.numel()

    avg_loss = total_loss / total_count
    print(f"Eval Loss: {avg_loss:.6f}")
    print(f"Perplexity: {np.exp(avg_loss):.6f}")

    Y_true = np.concatenate(Y_true, axis=0)
    Y_pred = np.concatenate(Y_pred, axis=0)
    pred_blockhit = np.argmax(Y_pred, axis=1)
    blockhit_true = Y_true.reshape(-1)
    accuracy = np.mean(blockhit_true == pred_blockhit)
    print(f"Accuracy: {accuracy:.6f}")

    if graph:
        plt.figure(figsize=(12, 6))
        plt.plot(blockhit_true, label="blockhit true", linewidth=1)
        plt.plot(pred_blockhit, label="blockhit pred", linewidth=1, alpha=0.8)
        plt.title("Blockhit Eval")
        plt.xlabel("sample")
        plt.ylabel("aUseItem")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return avg_loss


if __name__ == "__main__":
    blockhit_dataset = BlockhitDataset()
    train_windows = blockhit_dataset.get_train_windows(seq_len=SEQ_LEN)
    train_dataset = WindowDataset(train_windows)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if len(train_dataset) == 0:
        raise ValueError("No training windows found.")

    val_windows = blockhit_dataset.get_val_windows(seq_len=SEQ_LEN)
    val_dataset = WindowDataset(val_windows)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sample_x = train_dataset[0][0]
    input_dim = sample_x.numel()
    feature_dim = sample_x.shape[-1]
    output_dim = 2
    model_name = "stacked MLP 9"
    architecture = f"{input_dim} -> 512 -> 256 -> 128 -> 64 -> 2"

    model = BlockhitStackedMLP(input_dim=input_dim, output_dim=output_dim)
    # model = Blockhit1dCNN(input_dim=feature_dim, output_dim=output_dim)

    model_path = os.path.join(BASE_DIR, "blockhit_stacked_mlp_9.pth")
    checkpoint_meta = {
        "model_class": type(model).__name__,
        "model_name": model_name,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "architecture": architecture,
        "seq_len": SEQ_LEN,
    }
    train_info = train(
        model=model,
        dataloader=train_loader,
        val_loader=val_loader,
        save_path=model_path,
        checkpoint_meta=checkpoint_meta,
    )
    print(
        f"Training complete: "
        f"best_epoch={train_info['best_epoch']} "
        f"epochs_run={train_info['epochs_run']} "
        f"best_val_loss={train_info['best_val_loss']:.6f} "
        f"best_val_ppl={np.exp(train_info['best_val_loss']):.6f}"
    )

    eval_model(model=model, dataloader=val_loader, graph=True)
    print(f"Config: \n"
          f"  model: {model_name}\n"
          f"  input_dim: {input_dim}\n"
          f"  architecture: {architecture}\n"
          f"  seq_len: {SEQ_LEN}\n"
          # f"  batch_size: {BATCH_SIZE}\n"
          # f"  learning_rate: {LEARNING_RATE}\n"
          # f"  epochs: {EPOCHS}\n"
          f"  loss_type: CrossEntropy\n"
          f"  weights: {CLASS_WEIGHTS.numpy()}\n"
          # f"  best_epoch: {train_info['best_epoch']}\n"
          # f"  best_val_loss: {train_info['best_val_loss']:.6f}\n"
          # f"  best_val_ppl: {np.exp(train_info['best_val_loss']):.6f}\n"
    )

    print_class_distribution(train_dataset, "train")
    print_class_distribution(val_dataset, "val")
