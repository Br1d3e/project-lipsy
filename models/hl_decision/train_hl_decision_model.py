import os
from pathlib import Path

from sklearn.metrics import log_loss
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models.datasets import Datasets
from models.window_dataset import WindowDataset
from models.hl_decision.hl_decision_datasets import HLDecisionDataset, TARGETS, LABELS
from models.torch_dataset import TorchDataset

from matplotlib import pyplot as plt
import numpy as np
import torch
import xgboost as xgb


SEQ_LEN = 50
TRANSITION_CONTEXT = 40
LEARNING_RATE = 0.001
CLASS_WEIGHTS = torch.tensor((1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=torch.float32)
EPOCHS = 100
BATCH_SIZE = 128


class HLMultihead(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=128, output_dim=6):
        super().__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool1d(1),
        # )
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)

        self.heal_head = nn.Linear(hidden_dim, 1)
        self.pearl_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        h, _ = self.encoder(x)
        h = h[:, -1, :]

        heal_logits = self.heal_head(h)
        pearl_logits = self.pearl_head(h)
        aux_logits = torch.concat([heal_logits, pearl_logits], dim=-1)
        state_input = torch.concat([h, aux_logits], dim=-1)
        state_logits = self.state_head(state_input)
        return {
            "state_logits": state_logits,
            "heal_logits": heal_logits.squeeze(-1),
            "pearl_logits": pearl_logits.squeeze(-1),
        }


class TrainXgbModel:
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    @staticmethod
    def get_X_Y(dataset):
        if isinstance(dataset, tuple) and len(dataset) == 2:
            X, Y = dataset
            X = torch.as_tensor(X, dtype=torch.float32)
            if X.dim() > 2:
                X = X.flatten(start_dim=1)
            Y = torch.as_tensor(Y).long().view(-1)
            return X, Y

        if len(dataset) == 0:
            raise ValueError("empty dataset: no high-level decision windows were generated")

        X = []
        Y = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            X.append(x)
            Y.append(y)
        X = torch.stack(X).flatten(start_dim=1)
        Y = torch.stack(Y).long().view(-1)
        return X, Y

    def train(self, model, save_path=None):
        if not isinstance(model, xgb.XGBClassifier):
            raise ValueError(f"Invalid model: {model!r}")
        train_X, train_Y = self.get_X_Y(self.train_dataset)
        val_X, val_Y = self.get_X_Y(self.val_dataset)

        model.fit(train_X, train_Y, eval_set=[(val_X, val_Y)])

        print(f"Training Complete.")
        if model.early_stopping_rounds > 0:
            print(f"Best: #{model.best_iteration}, loss: {model.best_score:.6f}")
        print(f"Config: {model.get_xgb_params()}")

        if save_path is not None:
            model.save_model(save_path)
            print(f"Saved model to {save_path}")

        return model

    def eval(self, model):
        if not isinstance(model, xgb.XGBClassifier):
            raise ValueError(f"Invalid model: {model!r}")
        if not model.evals_result():
            raise ValueError(f"Invalid model evals: {model!r}")

        val_X, val_Y = self.get_X_Y(self.val_dataset)
        val_X_np = val_X.detach().cpu().numpy()
        val_Y_np = val_Y.detach().cpu().numpy().astype(np.int64)
        Y_prob = model.predict_proba(val_X_np)
        loss = log_loss(val_Y_np, Y_prob)
        ppl = np.exp(loss)
        print(f"Eval loss: {loss:.6f} | ppl: {ppl:.6f}")

        accuracy = model.score(val_X_np, val_Y_np)
        print(f"Accuracy: {accuracy:.6f}")

    @staticmethod
    def print_confusion_matrix(y_true, y_pred, module_name=""):
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")

        cm = np.zeros((2, 2), dtype=int)

        for t, p in zip(y_true, y_pred):
            if t not in (0, 1) or p not in (0, 1):
                continue
            cm[t, p] += 1

        print("\nConfusion matrix")
        print("rows = true, cols = pred")
        print("classes: 0, 1")
        print(cm)

        print("\nPretty:")
        print("              pred 0   pred 1")
        labels = ["true  0", "true  1"]
        for i in range(2):
            print(f"{labels[i]:>8} {cm[i, 0]:8d} {cm[i, 1]:8d}")

        print("\nPer-class precision/recall:")
        for c, name in enumerate(["0", "1"]):
            tp = cm[c, c]
            precision = tp / max(cm[:, c].sum(), 1)
            recall = tp / max(cm[c, :].sum(), 1)
            print(
                f"class {c} / hl decision {module_name} {name}: "
                f"precision={precision:.4f}, recall={recall:.4f}"
            )

class TrainTorchModel:
    def __init__(self, train_dataset, val_dataset, model=None, batch_size=BATCH_SIZE, train_sample_weights=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        if train_sample_weights is None:
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        else:
            train_sample_weights = np.asarray(train_sample_weights, dtype=np.float64)
            if train_sample_weights.shape != (len(self.train_dataset),):
                raise ValueError(
                    f"Expected {len(self.train_dataset)} train sample weights, "
                    f"got shape {train_sample_weights.shape}."
                )
            if not np.isfinite(train_sample_weights).all() or np.any(train_sample_weights < 0):
                raise ValueError("Train sample weights must be finite and non-negative.")
            if train_sample_weights.sum() <= 0:
                raise ValueError("Train sample weights must have a positive sum.")
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
                num_samples=len(self.train_dataset),
                replacement=True,
            )
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=train_sampler)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

    @staticmethod
    def save_model_checkpoint(model, save_path, checkpoint_meta=None):
        checkpoint = {
            "model_state_dict": model.state_dict(),
        }
        if checkpoint_meta:
            checkpoint.update(checkpoint_meta)
        torch.save(checkpoint, save_path)

    # def train(self, save_path=None, model=None, train_loader=None, val_loader=None, checkpoint_meta=None, class_weights=None,
    #           epochs=EPOCHS, lr=LEARNING_RATE, patience=25):
    #     if model is None:
    #         model = self.model
    #     if train_loader is None:
    #         train_loader = self.train_loader
    #     if val_loader is None:
    #         val_loader = self.val_loader
    #     device = self.get_device()
    #     model.to(device)
    #
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10)
    #     if class_weights is None:
    #         class_weights = CLASS_WEIGHTS
    #     class_weights = class_weights.to(device=device, dtype=torch.float32)
    #     criterion = nn.CrossEntropyLoss(weight=class_weights)
    #
    #     best_loss = 1e9
    #     best_epoch = -1
    #     best_state_dict = None
    #     loss_count = 0
    #     epochs_run = 0
    #     for epoch in range(epochs):
    #         epochs_run = epoch + 1
    #         total_loss = 0.0
    #         model.train()
    #
    #         for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
    #             xb, yb = xb.to(device), yb.to(device)
    #             y_class = self.binary_logits_to_class(yb).long()
    #             optimizer.zero_grad()
    #             pred = model(xb)
    #             loss = criterion(pred, y_class)
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #             optimizer.step()
    #             total_loss += loss.item()
    #
    #         avg_loss = total_loss / len(train_loader)
    #         # print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")
    #
    #         val_loss = self.eval(model, dataloader=val_loader)
    #         scheduler.step(val_loss)
    #
    #         print(
    #             f"Epoch {epoch + 1}/{epochs} | "
    #             f"Train Loss: {avg_loss:.6f} | "
    #             f"Val Loss: {val_loss:.6f} | "
    #             f"Val PPL: {np.exp(val_loss):.6f}"
    #         )
    #         if val_loss < best_loss:
    #             best_loss = val_loss
    #             best_epoch = epoch + 1
    #             best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    #             self.save_model_checkpoint(model, save_path, checkpoint_meta)
    #             loss_count = 0
    #             print("model saved")
    #         else:
    #             loss_count += 1
    #
    #         if loss_count > patience:
    #             break
    #
    #     if best_state_dict is not None:
    #         model.load_state_dict(best_state_dict)
    #     return {
    #         "best_val_loss": best_loss,
    #         "best_epoch": best_epoch,
    #         "epochs_run": epochs_run,
    #     }

    def train(self, save_path=None, model=None, train_loader=None, val_loader=None, checkpoint_meta=None, class_weights=None,
              epochs=EPOCHS, lr=LEARNING_RATE, patience=25):
        if model is None:
            model = self.model
        if train_loader is None:
            train_loader = self.train_loader
        if val_loader is None:
            val_loader = self.val_loader
        device = self.get_device()
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10)
        if class_weights is None:
            class_weights = CLASS_WEIGHTS
        class_weights = class_weights.to(device=device, dtype=torch.float32)
        state_criterion = nn.CrossEntropyLoss(weight=class_weights)
        aux_criterion = nn.BCEWithLogitsLoss()

        best_loss = 1e9
        best_epoch = -1
        best_state_dict = None
        loss_count = 0
        epochs_run = 0
        for epoch in range(epochs):
            epochs_run = epoch + 1
            total_loss = 0.0
            model.train()

            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                y_state = yb.reshape(-1).long()
                y_heal = (y_state == TARGETS["heal"]).float()
                y_pearl = (y_state == TARGETS["pearl"]).float()

                optimizer.zero_grad()
                pred = model(xb)
                pred_state = pred["state_logits"]
                pred_heal = pred["heal_logits"]
                pred_pearl = pred["pearl_logits"]

                state_loss = state_criterion(pred_state, y_state)
                heal_loss = aux_criterion(pred_heal, y_heal)
                pearl_loss = aux_criterion(pred_pearl, y_pearl)
                loss = state_loss + 0.5 * (heal_loss + pearl_loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            # print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f}")

            val_loss = self.eval(model, dataloader=val_loader)
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
                self.save_model_checkpoint(model, save_path, checkpoint_meta)
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

    def eval(self, model=None, graph=False, confusion_matrix=False, class_weights=None, module_name="",
             dataloader=None):
        if model is None:
            model = self.model
        if dataloader is None:
            dataloader = self.val_loader
        device = self.get_device()
        model.to(device)
        model.eval()

        if class_weights is None:
            class_weights = CLASS_WEIGHTS

        total_loss = 0.0
        total_count = 0
        class_weights = class_weights.to(device=device, dtype=torch.float32)
        state_criterion = nn.CrossEntropyLoss(weight=class_weights)
        aux_criterion = nn.BCEWithLogitsLoss()

        Y_true = []
        Y_pred = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                y_state = y.reshape(-1).long()
                y_heal = (y_state == TARGETS["heal"]).float()
                y_pearl = (y_state == TARGETS["pearl"]).float()

                pred = model(x)
                pred_state = pred["state_logits"]
                pred_heal = pred["heal_logits"]
                pred_pearl = pred["pearl_logits"]
                state_loss = state_criterion(pred_state, y_state)
                heal_loss = aux_criterion(pred_heal, y_heal)
                pearl_loss = aux_criterion(pred_pearl, y_pearl)
                loss = state_loss + 0.5 * (heal_loss + pearl_loss)

                Y_true.append(y_state.cpu().numpy())
                Y_pred.append(pred_state.cpu().numpy())

                total_loss += loss.item() * y_state.shape[0]
                total_count += y_state.shape[0]

        avg_loss = total_loss / total_count
        print(f"Eval Loss: {avg_loss:.6f}")
        print(f"Perplexity: {np.exp(avg_loss):.6f}")

        Y_true = np.concatenate(Y_true, axis=0)
        Y_pred = np.concatenate(Y_pred, axis=0)
        Y_class = self.multi_logits_to_class(Y_pred).detach().cpu().numpy()
        Y_true = Y_true.reshape(-1)
        accuracy = np.mean(Y_true == Y_class)
        print(f"Accuracy: {accuracy:.6f}")

        if confusion_matrix:
            self.print_confusion_matrix_multi_class(Y_true, Y_class)

        if graph:
            plt.figure(figsize=(12, 6))
            plt.plot(Y_true, label=f"HL {module_name} true", linewidth=1)
            plt.plot(Y_class, label=f"HL {module_name} pred", linewidth=1, alpha=0.8)
            plt.title(f"HL {module_name} Eval")
            plt.xlabel("sample")
            plt.ylabel("prediction class")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return avg_loss

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def binary_logits_to_class(logits, p_threshold=0.5):
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)
        if logits.dim() == 2 and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)
            return (probs[:, 1] > p_threshold).long()
        return torch.clamp(torch.round(logits.reshape(-1)).long(), 0, 1)

    @staticmethod
    def multi_logits_to_class(logits):
        if not torch.is_tensor(logits):
            logits = torch.as_tensor(logits)
        return torch.argmax(logits, dim=1).long()


    @staticmethod
    def print_confusion_matrix(y_eval, y_pred, module_name=""):
        TrainXgbModel.print_confusion_matrix(y_eval, y_pred, module_name)

    @staticmethod
    def print_confusion_matrix_multi_class(y_true, y_pred, n_classes=6):
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")
        if n_classes is None:
            n_classes = y_true.max() + 1

        cm = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(n_classes):
            for j in range(n_classes):
                true_mask = y_true == i
                pred_mask = y_pred == j
                cm[i, j] = np.sum(true_mask * pred_mask)

        print("\nConfusion matrix:")
        print(f"row: true, col: pred")
        print(f"classes: {np.arange(n_classes)}")
        print(cm)

        print("\nPer-class precision-recall:")
        for i in range(n_classes):
            precision = cm[i, i] / max(np.sum(cm[:, i]), 1)
            recall = cm[i, i] / max(np.sum(cm[i, :]), 1)
            print(f"class {i} ({LABELS[i]}): ")
            print(f"precision: {precision:.4f}, recall: {recall:.4f}")


if __name__ == "__main__":
    hl_dataset = HLDecisionDataset(use_cache=True, build_cache=False)
    train_distribution = [0.65, 0.15, 0.03, 0.05, 0.08, 0.04]
    train_x, train_y, sample_weights = hl_dataset.get_train_X_Y(
        seq_len=SEQ_LEN,
        return_sample_weights=True,
        expected_distribution=train_distribution,
    )
    print(train_x.shape, sample_weights.shape)
    train_dataset = TorchDataset(train_x, train_y)
    print("="*50)
    val_x, val_y = hl_dataset.get_val_X_Y(seq_len=SEQ_LEN)
    val_dataset = TorchDataset(val_x, val_y)

    input_dim = train_dataset[0][0].shape[-1]
    hl_model = HLMultihead(input_dim, 128, 6)
    trainer = TrainTorchModel(train_dataset, val_dataset, model=hl_model, train_sample_weights=sample_weights)

    save_path = Path(__file__).parent.resolve() / "models" / "hl_multihead_2.pt"
    train_info = trainer.train(save_path=save_path)
    print("Training Complete.")
    for info, data in train_info.items():
        print(f"  {info}: {data}")

    print(f"Config: \n"
          f"  model: \n{hl_model}\n"
          f"  train_distribution: {train_distribution}\n"
          f"  loss_type: CrossEntropy\n"
          f"  weights: {CLASS_WEIGHTS.numpy()}\n"
          )

    trainer.eval(hl_model, graph=True, confusion_matrix=True)
