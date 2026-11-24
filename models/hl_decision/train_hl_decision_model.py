from copy import deepcopy
from pathlib import Path

from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from models.datasets import Datasets
from models.window_dataset import WindowDataset
from models.hl_decision.hl_decision_datasets import (
    HLDecisionDataset, TARGETS, LABELS, FEATURE_TRANSFORMS, FEATURE_DEFAULTS, FEATURES
)
from models.torch_dataset import HLMultiTaskDataset

from matplotlib import pyplot as plt
import numpy as np
import torch
import xgboost as xgb

import shap

SHAP_BACKGROUND_SAMPLES = 64
SHAP_EVAL_SAMPLES = 128
SEQ_LEN = 100
TRANSITION_CONTEXT = 40
LEARNING_RATE = 0.001
CLASS_WEIGHTS = torch.tensor((1.0, 1.5, 3.0, 1.0, 1.0, 1.5), dtype=torch.float32)
EPOCHS = 100
BATCH_SIZE = 128
LOSS_WEIGHTS = {
    "state": 1.0,
    "heal": 0.5,
    "pearl": 0.5,
    "pot_soon": 0.25,
    "pot_urgency": 0.25,
}
AUXILIARY_TARGETS = {"heal": "heal_start", "pearl": "pearl_start",
                     "pot_soon": "target_pot_soon", "pot_urgency": "target_pot_urgency"}
POT_URGENCY_BETA = 1.0

R_HEAL_START = 126.09
R_PEARL_START = 455.8
HEAL_LOSS_WEIGHT = torch.as_tensor(0.1 * R_HEAL_START, dtype=torch.float32)
PEARL_LOSS_WEIGHT = torch.as_tensor(0.5 * R_PEARL_START, dtype=torch.float32)
HEAL_START_MULTIPLIER = 4
PEARL_START_MULTIPLIER = 4

class HLMultihead(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=128, output_dim=6, encoder_type="lstm"):
        super().__init__()
        self.model_config = dict(input_dim=input_dim, hidden_dim=hidden_dim,
                                 output_dim=output_dim, encoder_type=encoder_type)
        if encoder_type == "linear":
            self.encoder = nn.Linear(input_dim, hidden_dim)
        elif encoder_type == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        elif encoder_type == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown HL encoder: {encoder_type}")

        self.heal_head = nn.Linear(hidden_dim, 1)
        self.pearl_head = nn.Linear(hidden_dim, 1)
        self.pot_soon_head = nn.Linear(hidden_dim, 1)
        self.pot_urgency_head = nn.Linear(hidden_dim, 1)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        if isinstance(self.encoder, nn.LSTM):
            h, _ = self.encoder(x)
            h = h[:, -1, :]
        elif isinstance(self.encoder, nn.Linear):
            h = self.encoder(x[:, -1, :])
        else:
            h = self.encoder(x.transpose(1, 2))

        heal_logits = self.heal_head(h)
        pearl_logits = self.pearl_head(h)
        pot_soon_logits = self.pot_soon_head(h)
        pot_urgency_raw = self.pot_urgency_head(h)
        pot_urgency = torch.sigmoid(pot_urgency_raw)
        aux_predictions = torch.concat([
            torch.sigmoid(heal_logits),
            torch.sigmoid(pearl_logits),
            torch.sigmoid(pot_soon_logits),
            pot_urgency,
        ], dim=-1)

        state_input = torch.concat([h, aux_predictions], dim=-1)
        state_logits = self.state_head(state_input)
        return {
            "state_logits": state_logits,
            "heal_logits": heal_logits.squeeze(-1),
            "pearl_logits": pearl_logits.squeeze(-1),
            "pot_soon_logits": pot_soon_logits.squeeze(-1),
            "pot_urgency": pot_urgency.squeeze(-1),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        weights = checkpoint["model_state_dict"]
        config = checkpoint.get("model_config")
        if config is None:
            # Older checkpoints predate architecture metadata.
            if "encoder.weight_ih_l0" in weights:
                encoder_type, input_weight = "lstm", weights["encoder.weight_ih_l0"]
            elif "encoder.weight" in weights:
                encoder_type, input_weight = "linear", weights["encoder.weight"]
            else:
                encoder_type, input_weight = "cnn", weights["encoder.0.weight"]
            config = dict(input_dim=input_weight.shape[1],
                          hidden_dim=weights["heal_head.weight"].shape[1],
                          output_dim=weights["state_head.2.weight"].shape[0],
                          encoder_type=encoder_type)
        model = cls(**config)
        model.load_state_dict(weights)
        return model

class HLStateShapModel(nn.Module):
    """Expose state logits to SHAP without changing the multihead training interface."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        pred = self.model(x)
        return pred["state_logits"] if isinstance(pred, dict) else pred

class HLActionStartModel(nn.Module):
    """Binary action initiation classification model for benchmark"""
    def __init__(self, input_dim=65, hidden_dim=128, output_dim=1):
        super().__init__()
        if output_dim != 1:
            raise ValueError("HLActionStartModel requires one logit for BCEWithLogitsLoss")
        self.model_config = dict(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)

class HLMultiheadNoAux(nn.Module):
    """State prediction model without auxiliary targets"""
    def __init__(self, input_dim=65, hidden_dim=128, output_dim=6, encoder_type="lstm"):
        super().__init__()
        self.model_config = dict(input_dim=input_dim, hidden_dim=hidden_dim,
                                 output_dim=output_dim, encoder_type=encoder_type)
        if encoder_type == "linear":
            self.encoder = nn.Linear(input_dim, hidden_dim)
        elif encoder_type == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        elif encoder_type == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown HL encoder: {encoder_type}")

        self.state_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if isinstance(self.encoder, nn.LSTM):
            h, _ = self.encoder(x)
            h = h[:, -1, :]
        elif isinstance(self.encoder, nn.Linear):
            h = self.encoder(x[:, -1, :])
        else:
            h = self.encoder(x.transpose(1, 2))

        return self.state_head(h)

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
    def __init__(self, train_dataset=None, val_dataset=None, model=None, batch_size=BATCH_SIZE, train_sample_weights=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        if train_dataset is None:
            self.train_loader = None
        else:
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
        if self.val_dataset is not None:
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

    @staticmethod
    def save_model_checkpoint(model, save_path, checkpoint_meta=None):
        checkpoint = {
            "model_state_dict": model.state_dict(),
        }
        if isinstance(model, (HLMultihead, HLMultiheadNoAux, HLActionStartModel)):
            checkpoint["model_type"] = type(model).__name__
            checkpoint["model_config"] = dict(model.model_config)
        if checkpoint_meta:
            checkpoint.update(checkpoint_meta)
        torch.save(checkpoint, save_path)

    @staticmethod
    def initiation_possible(state_target, action_start, action):
        """Loss/metric eligibility from t+1 labels; not a runtime availability check."""
        return (state_target.reshape(-1) != TARGETS[action]) | action_start.reshape(-1).bool()

    @staticmethod
    def masked_initiation_loss(logits, starts, state_target, action, criterion):
        possible = TrainTorchModel.initiation_possible(state_target, starts, action)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, starts, reduction="none",
            weight=criterion.weight.to(logits.device) if criterion.weight is not None else None,
            pos_weight=criterion.pos_weight.to(logits.device) if hasattr(criterion, "pos_weight") and criterion.pos_weight is not None else None,
        )
        return (losses * possible).sum() / possible.sum().clamp_min(1)

    @staticmethod
    def resolve_auxiliary_heads(auxiliary_heads=None, model=None):
        """Select supervised heads; [] disables auxiliaries without changing the architecture."""
        if auxiliary_heads is None:
            auxiliary_heads = getattr(model, "checkpoint_meta", {}).get("auxiliary_heads")
        if auxiliary_heads is None:
            auxiliary_heads = () if isinstance(model, HLMultiheadNoAux) else tuple(AUXILIARY_TARGETS)
        heads = tuple(auxiliary_heads)
        if len(set(heads)) != len(heads) or set(heads) - AUXILIARY_TARGETS.keys():
            raise ValueError(f"Invalid auxiliary heads: {heads}")
        if isinstance(model, HLMultiheadNoAux) and heads:
            raise ValueError("HLMultiheadNoAux has no auxiliary outputs")
        return heads

    @staticmethod
    def loss_sample_counts(targets, auxiliary_heads=None):
        heads = TrainTorchModel.resolve_auxiliary_heads(auxiliary_heads)
        counts = {name: targets["state"].numel() for name in ("state", *heads)}
        for action in set(heads) & {"heal", "pearl"}:
            counts[action] = int(TrainTorchModel.initiation_possible(
                targets["state"], targets[f"{action}_start"], action,
            ).sum().item())
        return counts

    @staticmethod
    def compute_losses(pred, targets, state_criterion, binary_criterion, urgency_criterion,
                       loss_weights=LOSS_WEIGHTS, class_weights=CLASS_WEIGHTS, heal_criterion=None, pearl_criterion=None,
                       heal_start_weight=HEAL_START_MULTIPLIER, pearl_start_weight=PEARL_START_MULTIPLIER, state_only=False, auxiliary_heads=None):
        heads = TrainTorchModel.resolve_auxiliary_heads(() if state_only else auxiliary_heads)
        required_targets = {"state", "heal_start", "pearl_start"} | {AUXILIARY_TARGETS[h] for h in heads}
        missing_targets = required_targets.difference(targets)
        if missing_targets:
            raise ValueError(f"Missing multitask targets: {sorted(missing_targets)}")

        if heal_criterion is None:
            heal_criterion = BCEWithLogitsLoss(pos_weight=HEAL_LOSS_WEIGHT)

        if pearl_criterion is None:
            pearl_criterion = BCEWithLogitsLoss(pos_weight=PEARL_LOSS_WEIGHT)

        y_state = targets["state"].reshape(-1).long()
        y_heal = targets["heal_start"].reshape(-1).float()
        y_pearl = targets["pearl_start"].reshape(-1).float()

        state_loss = state_criterion(pred["state_logits"] if isinstance(pred, dict) else pred, y_state)
        start_weights = torch.ones_like(state_loss, dtype=torch.float32)
        start_weights[y_heal.bool()] = heal_start_weight
        start_weights[y_pearl.bool()] = pearl_start_weight
        state_loss_weighted = (
                (state_loss * start_weights).sum() /
                (start_weights * class_weights[y_state]).sum()
        )

        losses = {"state": state_loss_weighted}
        for name in heads:
            y_true = targets[AUXILIARY_TARGETS[name]].reshape(-1).float()
            if name in ("heal", "pearl"):
                criterion = heal_criterion if name == "heal" else pearl_criterion
                losses[name] = TrainTorchModel.masked_initiation_loss(
                    pred[f"{name}_logits"], y_true, y_state, name, criterion)
            elif name == "pot_soon":
                losses[name] = binary_criterion(pred["pot_soon_logits"], y_true)
            else:
                losses[name] = urgency_criterion(pred["pot_urgency"], y_true)
        total_loss = sum(loss_weights[name] * loss for name, loss in losses.items())
        return total_loss, losses, y_state

    @staticmethod
    def compute_start_losses(pred, targets, action_name="heal_start", start_criterion=None):
        if action_name not in ("heal_start", "pearl_start"):
            raise ValueError(f"Unknown action name: {action_name!r}")
        action = action_name.removesuffix("_start")
        if start_criterion is None:
            weight = HEAL_LOSS_WEIGHT if action == "heal" else PEARL_LOSS_WEIGHT
            start_criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        if not isinstance(start_criterion, nn.BCEWithLogitsLoss):
            raise TypeError("Independent action starts require BCEWithLogitsLoss")

        y_state = targets["state"].reshape(-1).long()
        y_start = targets[action_name].reshape(-1).float()
        if pred.shape not in (y_start.shape, (len(y_start), 1)):
            raise ValueError("Expected one action-start logit per sample")
        return TrainTorchModel.masked_initiation_loss(
            pred.reshape(-1), y_start, y_state, action, start_criterion,
        )

    @staticmethod
    def sequence_feature_names(seq_len, features):
        return [f"t-{seq_len - 1 - tick}:{feature}" for tick in range(seq_len) for feature in features]

    def train(self, model_name="hl_multihead.pt", save_path=None, model=None, train_loader=None,
              val_loader=None, checkpoint_meta=None, class_weights=None,
              epochs=EPOCHS, lr=LEARNING_RATE, patience=25, auxiliary_heads=None):
        if model is None:
            model = self.model
        heads = self.resolve_auxiliary_heads(auxiliary_heads, model)
        if train_loader is None:
            train_loader = self.train_loader
        if val_loader is None:
            val_loader = self.val_loader
        base_path = Path(save_path) if save_path is not None else Path(__file__).resolve().parent / "models" / model_name
        base_path.parent.mkdir(parents=True, exist_ok=True)
        paths = {name: base_path.with_name(f"{base_path.stem}_{name}.pt")
                 for name in ("loss", "heal", "pearl")}

        device = self.get_device()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10)
        if class_weights is None:
            class_weights = CLASS_WEIGHTS
        class_weights = class_weights.to(device=device, dtype=torch.float32)
        state_criterion_per_sample = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        binary_criterion = nn.BCEWithLogitsLoss()
        urgency_criterion = nn.SmoothL1Loss(beta=POT_URGENCY_BETA)
        checkpoint_meta = dict(checkpoint_meta or {})
        checkpoint_meta.update({
            "model_config": dict(model.model_config),
            "auxiliary_heads": list(heads),
            "model_architecture": str(model),
            "training_config": {
                "torch_seed": torch.initial_seed(), "lr": lr, "epochs": epochs,
                "patience": patience, "batch_size": train_loader.batch_size,
                "sampler": type(train_loader.sampler).__name__,
                "optimizer": "AdamW", "weight_decay": optimizer.defaults["weight_decay"],
                "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.7, "patience": 10},
                "class_weights": class_weights.cpu().tolist(),
                "loss_weights": {name: LOSS_WEIGHTS[name] for name in ("state", *heads)},
                "heal_pos_weight": HEAL_LOSS_WEIGHT.item(), "pearl_pos_weight": PEARL_LOSS_WEIGHT.item(),
                "heal_start_multiplier": HEAL_START_MULTIPLIER,
                "pearl_start_multiplier": PEARL_START_MULTIPLIER,
                "pot_urgency_beta": POT_URGENCY_BETA,
                "initiation_loss_mask": "state[t+1] != action OR action_start[t+1]; mean over eligible frames",
            },
            "initiation_metric": "label_start truth; main argmax rising edges vs auxiliary >= 0.5; contiguous endpoints only",
        })

        model.checkpoint_meta = checkpoint_meta
        checkpoint_heads = {name: "auxiliary" if name in heads else "state"
                            for name in ("heal", "pearl")}
        best = {"loss": float("inf"), "heal": -1.0, "pearl": -1.0}
        best_epochs = {}
        best_metrics = {}
        best_state_dict = None
        stale_epochs = 0
        epochs_run = 0
        for epoch in range(epochs):
            epochs_run = epoch + 1
            total_count = 0
            component_totals = {name: 0.0 for name in ("state", *heads)}
            component_counts = {name: 0 for name in ("state", *heads)}
            model.train()
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                xb = xb.to(device)
                yb = {key: value.to(device) for key, value in yb.items()}
                optimizer.zero_grad()
                pred = model(xb)
                loss, losses, _ = self.compute_losses(
                    pred, yb, state_criterion_per_sample, binary_criterion, urgency_criterion,
                    class_weights=class_weights, auxiliary_heads=heads,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_size = xb.shape[0]
                total_count += batch_size
                counts = self.loss_sample_counts(yb, heads)
                for name, component_loss in losses.items():
                    component_totals[name] += component_loss.item() * counts[name]
                    component_counts[name] += counts[name]

            component_means = {k: v / max(component_counts[k], 1) for k, v in component_totals.items()}
            train_loss = sum(LOSS_WEIGHTS[k] * v for k, v in component_means.items())
            eval_results = self.eval(model, dataloader=val_loader, inspect_initiation=True,
                                     class_weights=class_weights, auxiliary_heads=heads)
            val_loss = eval_results["avg_loss"]
            scheduler.step(val_loss)
            scores = {"loss": val_loss}
            scores.update({name: eval_results["initiation"][head][name]["f1"]
                           for name, head in checkpoint_heads.items()})
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            improved = False
            for name, score in scores.items():
                is_better = score < best[name] if name == "loss" else score > best[name]
                if not is_better:
                    continue
                improved = True
                best[name] = score
                best_epochs[name] = epoch + 1
                best_metrics[name] = deepcopy(eval_results)
                self.save_model_checkpoint(model, paths[name], {
                    **checkpoint_meta, "epoch": epoch + 1, "selection_criterion": name,
                    "selection_metric": "avg_loss" if name == "loss" else f"initiation.{checkpoint_heads[name]}.{name}.f1",
                    "selection_score": score, "validation_metrics": eval_results,
                    "train_metrics": {"avg_loss": train_loss, "components": component_means,
                                      "loss_sample_counts": component_counts, "sample_count": total_count},
                })
                if name == "loss":
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"Saved best {name} ({checkpoint_heads.get(name, 'loss')}): epoch {epoch + 1}, score {score:.6f} -> {paths[name]}")
            stale_epochs = 0 if improved else stale_epochs + 1
            if not improved and stale_epochs >= patience:
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        return {
            "best_val_loss": best["loss"], "best_heal_f1": best["heal"],
            "best_pearl_f1": best["pearl"], "best_epoch": best_epochs.get("loss"),
            "best_epochs": best_epochs, "best_metrics": best_metrics,
            "checkpoint_paths": {k: str(v) for k, v in paths.items()},
            "restored_criterion": "loss", "epochs_run": epochs_run,
        }

    def eval(self, model=None, graph=False, confusion_matrix=False, inspect_initiation=False, interpret_shap=False, class_weights=None, module_name="",
             dataloader=None, test_X=None, auxiliary_heads=None):
        if model is None:
            model = self.model
        heads = self.resolve_auxiliary_heads(auxiliary_heads, model)
        if dataloader is None:
            dataloader = self.val_loader
        device = self.get_device()
        model.to(device)
        model.eval()

        if class_weights is None:
            class_weights = CLASS_WEIGHTS

        total_count = 0
        component_totals = {name: 0.0 for name in ("state", *heads)}
        component_counts = {name: 0 for name in ("state", *heads)}
        class_weights = class_weights.to(device=device, dtype=torch.float32)
        # state_criterion = nn.CrossEntropyLoss(weight=class_weights)
        state_criterion_per_sample = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        binary_criterion = nn.BCEWithLogitsLoss()
        urgency_criterion = nn.SmoothL1Loss(beta=POT_URGENCY_BETA)

        Y_true = []
        Y_pred = []
        sample_identity = {"round_id": [], "target_tick": []}
        auxiliary_true = {name: [] for name in set(heads) | {"heal", "pearl"}}
        auxiliary_pred = {name: [] for name in heads}

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = {key: value.to(device) for key, value in y.items()}

                pred = model(x)
                pred_state = pred["state_logits"] if isinstance(pred, dict) else pred
                loss, losses, y_state = self.compute_losses(
                    pred,
                    y,
                    state_criterion_per_sample,
                    binary_criterion,
                    urgency_criterion,
                    class_weights=class_weights, auxiliary_heads=heads,
                )

                if inspect_initiation:
                    for key in sample_identity:
                        if key not in y:
                            raise ValueError(f"Initiation evaluation requires {key} endpoint metadata; regenerate windows.")
                        sample_identity[key].append(y[key].reshape(-1).cpu().numpy())
                Y_true.append(y_state.cpu().numpy())
                Y_pred.append(pred_state.cpu().numpy())
                for name in auxiliary_true:
                    auxiliary_true[name].append(y[AUXILIARY_TARGETS[name]].reshape(-1).cpu().numpy())
                for name in heads:
                    value = pred["pot_urgency"] if name == "pot_urgency" else torch.sigmoid(pred[f"{name}_logits"])
                    auxiliary_pred[name].append(value.cpu().numpy())

                counts = self.loss_sample_counts(y, heads)
                for name, component_loss in losses.items():
                    component_totals[name] += component_loss.item() * counts[name]
                    component_counts[name] += counts[name]
                total_count += y_state.shape[0]

        component_means = {k: v / max(component_counts[k], 1) for k, v in component_totals.items()}
        avg_loss = sum(LOSS_WEIGHTS[k] * v for k, v in component_means.items())
        print(f"Eval Loss: {avg_loss:.6f}")
        print(
            "Eval components: "
            + " | ".join(
                f"{name}: {value:.6f}"
                for name, value in component_means.items()
            )
        )

        Y_true = np.concatenate(Y_true, axis=0)
        Y_pred = np.concatenate(Y_pred, axis=0)
        auxiliary_true = {
            name: np.concatenate(values).reshape(-1)
            for name, values in auxiliary_true.items()
        }
        auxiliary_pred = {
            name: np.concatenate(values).reshape(-1)
            for name, values in auxiliary_pred.items()
        }
        Y_class = self.multi_logits_to_class(Y_pred).detach().cpu().numpy()
        Y_true = Y_true.reshape(-1)
        accuracy = np.mean(Y_true == Y_class)
        print(f"Accuracy: {accuracy:.6f}")

        initiation_masks = {
            name: self.initiation_possible(
                torch.as_tensor(Y_true), torch.as_tensor(auxiliary_true[name]), name
            ).numpy()
            for name in heads if name in ("heal", "pearl")
        }
        loss_coverage = {}
        for name in initiation_masks:
            loss_coverage[name] = {
                "eligible_count": component_counts[name],
                "excluded_continuation_count": total_count - component_counts[name],
                "retained_start_count": int(auxiliary_true[name].sum()),
            }
            print(f"{name} initiation loss mask: {loss_coverage[name]}")
        binary_metrics = {}
        for name in initiation_masks:
            binary_metrics[name] = self.print_binary_head_metrics(
                name,
                auxiliary_true[name][initiation_masks[name]],
                auxiliary_pred[name][initiation_masks[name]],
                component_means[name],
            )
        if "pot_soon" in heads:
            binary_metrics["pot_soon"] = self.print_binary_head_metrics(
                "pot_soon", auxiliary_true["pot_soon"], auxiliary_pred["pot_soon"], component_means["pot_soon"])
        urgency_metrics = None
        if "pot_urgency" in heads:
            urgency_metrics = self.print_regression_head_metrics(
                "pot_urgency", auxiliary_true["pot_urgency"], auxiliary_pred["pot_urgency"], component_means["pot_urgency"])

        if confusion_matrix:
            self.print_confusion_matrix_multi_class(Y_true, Y_class)

        if graph:
            self.plot_multitask_eval(
                Y_true,
                Y_class,
                auxiliary_true,
                auxiliary_pred,
                binary_metrics,
                module_name,
                initiation_masks=initiation_masks,
            )

        if interpret_shap:
            if test_X is None or len(test_X) == 0:
                raise ValueError("Non-empty test_X must be provided for SHAP interpretation.")

            model_cpu = HLStateShapModel(deepcopy(model).cpu()).eval()
            background_X = torch.as_tensor(test_X[:min(SHAP_BACKGROUND_SAMPLES, len(test_X))], dtype=torch.float32)
            explain_X = torch.as_tensor(test_X[:min(SHAP_EVAL_SAMPLES, len(test_X))], dtype=torch.float32)

            explainer = shap.GradientExplainer(model_cpu, data=background_X)
            shap_values = explainer.shap_values(explain_X)

            for cls in range(6):
                if isinstance(shap_values, list):
                    class_shap = shap_values[cls]
                elif shap_values.ndim == explain_X.ndim + 1:
                    class_shap = shap_values[..., cls]
                else:
                    class_shap = shap_values
                shap.summary_plot(
                    class_shap.reshape(class_shap.shape[0], -1),
                    explain_X.numpy().reshape(explain_X.shape[0], -1),
                    feature_names=self.sequence_feature_names(test_X.shape[1], FEATURES),
                )

        if inspect_initiation:
            identity = {key: np.concatenate(values) for key, values in sample_identity.items()}
            initiation = self.score_initiation(Y_class, auxiliary_true, auxiliary_pred, identity, initiation_masks)
            return {
                "avg_loss": avg_loss, "accuracy": float(accuracy), "sample_count": total_count,
                "components": component_means, "initiation_loss_coverage": loss_coverage,
                "state_confusion_matrix": np.bincount(
                    Y_true.astype(int) * len(LABELS) + Y_class, minlength=len(LABELS) ** 2
                ).reshape(len(LABELS), len(LABELS)).tolist(),
                "binary_metrics": {k: {m: v for m, v in result.items() if m != "y_pred"}
                                   for k, result in binary_metrics.items()},
                "urgency_metrics": urgency_metrics, "initiation": initiation,
                "heal_initiation_f1": initiation["state"]["heal"]["f1"],
                "pearl_initiation_f1": initiation["state"]["pearl"]["f1"],
            }
        return avg_loss

    def train_action_start(self, model_name="hl_multihead.pt", action_name="heal_start", save_path=None, model=None, train_loader=None,
              val_loader=None, checkpoint_meta=None,
              epochs=EPOCHS, lr=LEARNING_RATE, patience=25):
        if model is None:
            model = self.model
        if train_loader is None:
            train_loader = self.train_loader
        if val_loader is None:
            val_loader = self.val_loader
        path = Path(save_path) if save_path is not None else Path(
            __file__).resolve().parent / "models" / model_name
        path.parent.mkdir(parents=True, exist_ok=True)

        device = self.get_device()
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10)
        checkpoint_meta = dict(checkpoint_meta or {})
        checkpoint_meta.update({
            "model_config": dict(model.model_config),
            "action_name": action_name,
            "output_dim": 1,
            "model_architecture": str(model),
            "training_config": {
                "torch_seed": torch.initial_seed(), "lr": lr, "epochs": epochs,
                "loss_type": "BCEWithLogitsLoss",
                "patience": patience, "batch_size": train_loader.batch_size,
                "sampler": type(train_loader.sampler).__name__,
                "optimizer": "AdamW", "weight_decay": optimizer.defaults["weight_decay"],
                "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.7, "patience": 10},
                "heal_pos_weight": HEAL_LOSS_WEIGHT.item(), "pearl_pos_weight": PEARL_LOSS_WEIGHT.item(),
                "initiation_loss_mask": "state[t+1] != action OR action_start[t+1]; mean over eligible frames",
            },
            "initiation_metric": "label_start truth; sigmoid(logit) >= 0.5; all endpoints",
        })

        best_state_dict = None
        best_epoch = 0
        best_loss = float("inf")
        stale_epochs = 0
        epochs_run = 0
        for epoch in range(epochs):
            epochs_run = epoch + 1
            total_count = 0
            total_loss = 0.0
            model.train()
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
                xb = xb.to(device)
                yb = {key: value.to(device) for key, value in yb.items()}
                optimizer.zero_grad()
                pred = model(xb)
                loss = self.compute_start_losses(
                    pred, yb,
                    action_name=action_name
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                count = self.initiation_possible(yb["state"], yb[action_name], action_name.removesuffix("_start")).sum().item()
                total_count += count
                total_loss += loss.item() * count

            train_loss = total_loss / max(total_count, 1)
            val_loss = self.eval_action_start(model, action_name=action_name, dataloader=val_loader)
            scheduler.step(val_loss)
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch + 1
                stale_epochs = 0
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                self.save_model_checkpoint(model, path, {
                    **checkpoint_meta, "epoch": epoch + 1, "validation_loss": val_loss,
                })
                print(f"Saved best model: epoch {epoch + 1}")
            else:
                stale_epochs += 1

            if stale_epochs >= patience:
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            model.checkpoint_meta = {**checkpoint_meta, "epoch": best_epoch, "validation_loss": best_loss}
        return {
            "best_val_loss": best_loss,
            "best_epoch": best_epoch,
            "checkpoint_path": path,
            "restored_criterion": "loss", "epochs_run": epochs_run,
        }

    def eval_action_start(self, model=None, action_name="heal_start", graph=False, confusion_matrix=False, dataloader=None):
        if model is None:
            model = self.model
        if dataloader is None:
            dataloader = self.val_loader
        device = self.get_device()
        model.to(device)
        model.eval()

        Y_true = []
        Y_pred = []
        eligible_masks = []
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)
                y = {key: value.to(device) for key, value in y.items()}

                y_action = y[action_name].reshape(-1)
                pred = model(x)
                loss = self.compute_start_losses(
                    pred,
                    y,
                    action_name=action_name
                )

                Y_true.append(y_action.cpu().numpy())
                Y_pred.append(torch.sigmoid(pred.reshape(-1)).cpu().numpy())
                eligible = self.initiation_possible(y["state"], y_action, action_name.removesuffix("_start"))
                eligible_masks.append(eligible.cpu().numpy())
                count = eligible.sum().item()
                total_loss += loss.item() * count
                total_count += count

        avg_loss = total_loss / max(total_count, 1)
        print(f"Eval Loss: {avg_loss:.6f}")

        Y_true = np.concatenate(Y_true, axis=0)
        Y_pred = np.concatenate(Y_pred, axis=0)
        eligible = np.concatenate(eligible_masks)
        print(f"{action_name} initiation mask: {eligible.sum()}/{len(eligible)} eligible frames")
        Y_true = Y_true[eligible]
        Y_pred = Y_pred[eligible]
        Y_class = Y_pred >= 0.5
        accuracy = float(np.mean(Y_true == Y_class)) if len(Y_true) else float("nan")
        print(f"Accuracy: {accuracy:.6f}")

        self.print_binary_head_metrics(
            action_name,
            Y_true,
            Y_pred,
            avg_loss,
        )

        if confusion_matrix:
            self.print_confusion_matrix(Y_true, Y_class, action_name)

        if graph:
            self.plot_binary_eval(Y_true, Y_pred, action_name)

        return avg_loss

    @staticmethod
    def score_initiation(state_predictions, start_targets, auxiliary_probabilities, identity, initiation_masks=None):
        rounds = np.asarray(identity["round_id"]).reshape(-1)
        ticks = np.asarray(identity["target_tick"]).reshape(-1)
        states = np.asarray(state_predictions).reshape(-1)
        if rounds.shape != states.shape or ticks.shape != states.shape:
            raise ValueError("Initiation endpoint metadata must align with predictions")
        # No transition can be inferred at the first endpoint or across a gap/round.
        contiguous = np.zeros(len(states), dtype=bool)
        contiguous[1:] = (rounds[1:] == rounds[:-1]) & (ticks[1:] == ticks[:-1] + 1)
        result = {"state": {}, "auxiliary": {}, "coverage": {
            "total_samples": len(states), "evaluated_samples": int(contiguous.sum()),
            "excluded_samples": int((~contiguous).sum()),
        }}
        print(f"\nInitiation comparison: {contiguous.sum()}/{len(states)} contiguous endpoints")
        for name in ("heal", "pearl"):
            truth = np.asarray(start_targets[name]).reshape(-1)
            probabilities = np.asarray(auxiliary_probabilities[name]).reshape(-1) if name in auxiliary_probabilities else None
            if truth.shape != states.shape or (probabilities is not None and probabilities.shape != states.shape):
                raise ValueError(f"{name} start targets/predictions must align with endpoints")
            active = states == TARGETS[name]
            starts = np.zeros(len(states), dtype=bool)
            starts[1:] = active[1:] & ~active[:-1]
            result["coverage"][f"excluded_{name}_starts"] = int(truth[~contiguous].sum())
            for head, prediction in (("state", starts), ("auxiliary", probabilities)):
                if prediction is None:
                    continue
                eligible = contiguous
                if head == "auxiliary" and initiation_masks is not None:
                    eligible = contiguous & initiation_masks[name]
                metrics = TrainTorchModel.print_binary_head_metrics(
                    f"{name} {head} initiation", truth[eligible], prediction[eligible],
                    loss=None,
                )
                result[head][name] = {k: v for k, v in metrics.items() if k != "y_pred"}
            print(f"Excluded {name} starts at boundaries/gaps: {result['coverage'][f'excluded_{name}_starts']}")
        return result

    @staticmethod
    def print_binary_head_metrics(name, y_true, y_probability, loss, threshold=0.5):
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_probability = np.asarray(y_probability, dtype=np.float64).reshape(-1)
        if y_true.shape != y_probability.shape:
            raise ValueError(
                f"{name} target/prediction shape mismatch: {y_true.shape} vs {y_probability.shape}"
            )
        if not np.isfinite(y_probability).all():
            raise ValueError(f"{name} predictions contain non-finite values")

        y_pred = (y_probability >= threshold).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        accuracy = (tp + tn) / max(len(y_true), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        average_precision = (
            average_precision_score(y_true, y_probability)
            if np.unique(y_true).size == 2
            else float("nan")
        )

        print(f"\n{name} head:")
        loss_text = f"loss={loss:.6f} | " if loss is not None else ""
        print(f"{loss_text}threshold={threshold:.2f} | accuracy={accuracy:.6f}")
        print(
            f"precision={precision:.6f} | recall={recall:.6f} | "
            f"f1={f1:.6f} | average_precision={average_precision:.6f}"
        )
        print(
            f"true positives={int(y_true.sum())}/{len(y_true)} | "
            f"pred positives={int(y_pred.sum())}/{len(y_pred)}"
        )
        print(f"confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

        return {
            "loss": float(loss) if loss is not None else None, "threshold": threshold, "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1,
            "average_precision": average_precision, "y_pred": y_pred,
            "sample_count": len(y_true), "positive_count": int(y_true.sum()),
            "predicted_positive_count": int(y_pred.sum()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }

    @staticmethod
    def print_regression_head_metrics(name, y_true, y_pred, loss):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if y_true.shape != y_pred.shape:
            raise ValueError(f"{name} target/prediction shape mismatch: {y_true.shape} vs {y_pred.shape}")
        if not np.isfinite(y_pred).all():
            raise ValueError(f"{name} predictions contain non-finite values")

        error = y_pred - y_true
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error ** 2))
        correlation = (
            np.corrcoef(y_true, y_pred)[0, 1]
            if np.std(y_true) > 0.0 and np.std(y_pred) > 0.0
            else float("nan")
        )
        print(f"\n{name} head:")
        print(
            f"loss={loss:.6f} | MAE={mae:.6f} | RMSE={rmse:.6f} | "
            f"correlation={correlation:.6f}"
        )
        print(
            f"true mean/std={np.mean(y_true):.6f}/{np.std(y_true):.6f} | "
            f"pred mean/std={np.mean(y_pred):.6f}/{np.std(y_pred):.6f}"
        )

        return {"loss": float(loss), "mae": float(mae), "rmse": float(rmse),
                "correlation": float(correlation), "true_mean": float(np.mean(y_true)),
                "true_std": float(np.std(y_true)), "pred_mean": float(np.mean(y_pred)),
                "pred_std": float(np.std(y_pred))}

    @staticmethod
    def plot_multitask_eval(y_state, pred_state, auxiliary_true, auxiliary_pred,
                            binary_metrics, module_name="", initiation_masks=None):
        title_prefix = f"HL {module_name}" if module_name else "HL"
        names = list(auxiliary_pred)
        figure, axes = plt.subplots(1 + len(names), 1, figsize=(14, 3 * (1 + len(names))), sharex=True, squeeze=False)
        axes = axes[:, 0]
        axes[0].plot(y_state, label="true", linewidth=1)
        axes[0].plot(pred_state, label="pred", linewidth=1, alpha=0.8)
        axes[0].set_title(f"{title_prefix} state")
        axes[0].set_ylabel("class")

        for axis, name in zip(axes[1:], names):
            axis.plot(auxiliary_true[name], label="true", linewidth=1)
            axis.plot(auxiliary_pred[name], label="probability" if name != "pot_urgency" else "pred",
                      linewidth=1, alpha=0.8)
            axis.set_title(f"{title_prefix} {name}")
            axis.set_ylabel("value")

        axes[-1].set_xlabel("sample")
        for axis in axes:
            axis.legend()
        figure.tight_layout()

        pr_figure, pr_axis = plt.subplots(figsize=(8, 6))
        plotted_pr_curve = False
        for name in binary_metrics:
            eligible = initiation_masks[name] if initiation_masks is not None and name in initiation_masks else slice(None)
            y_true = auxiliary_true[name][eligible]
            if np.unique(y_true).size < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_true, auxiliary_pred[name][eligible])
            ap = binary_metrics[name]["average_precision"]
            pr_axis.plot(recall, precision, label=f"{name} AP={ap:.3f}")
            plotted_pr_curve = True
        pr_axis.set_title(f"{title_prefix} auxiliary precision-recall")
        pr_axis.set_xlabel("recall")
        pr_axis.set_ylabel("precision")
        pr_axis.set_xlim(0.0, 1.0)
        pr_axis.set_ylim(0.0, 1.05)
        if plotted_pr_curve:
            pr_axis.legend()
        pr_figure.tight_layout()
        plt.show()


    @staticmethod
    def plot_binary_eval(y, pred, module_name=""):
        title_prefix = f"HL {module_name}" if module_name else "HL"

        plt.plot(y, label="true", linewidth=1)
        plt.plot(pred, label="probability", linewidth=1, alpha=0.8)
        plt.title(title_prefix)
        plt.ylabel("value")
        plt.xlabel("sample")

        plt.show()

        if np.unique(y).size < 2:
            return
        precision, recall, _ = precision_recall_curve(y, pred)
        ap = average_precision_score(y, pred)
        plt.plot(recall, precision, label=f"{module_name} AP={ap:.3f}")
        plt.title(f"{title_prefix} auxiliary precision-recall")
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.show()

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
        return (torch.sigmoid(logits.reshape(-1)) >= p_threshold).long()

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

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def load_model(path, device=None, return_transforms=False):
        if device is None:
            device = TrainTorchModel.get_device()

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        feature_transforms = None
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            feature_transforms = checkpoint.get("feature_transforms", None)
            if checkpoint.get("model_type") in ("HLActionStartModel", "HLMultiheadNoAux"):
                model_class = HLActionStartModel if checkpoint["model_type"] == "HLActionStartModel" else HLMultiheadNoAux
                model = model_class(**checkpoint["model_config"])
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model = HLMultihead.from_checkpoint(checkpoint)
        else:
            model = checkpoint
        model.to(device)
        model.eval()
        model.checkpoint_meta = checkpoint

        if feature_transforms is None and return_transforms:
            raise ValueError("feature_transforms not available")

        if return_transforms:
            return model, feature_transforms

        return model


def train_eval_model(model_name="hl_multihead_?.pt", hl_model=None, seed=42, mode="multitask", action_name="heal_start", auxiliary_heads=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    hl_dataset = HLDecisionDataset(use_cache=True, build_cache=False)
    # train_distribution = [0.65, 0.15, 0.03, 0.05, 0.08, 0.04]
    train_x, train_y = hl_dataset.get_train_X_Y(
        seq_len=SEQ_LEN,
        return_auxiliary_targets=True,
        # return_sample_weights=True,
        # expected_distribution=train_distribution,
    )
    train_dataset = HLMultiTaskDataset(train_x, train_y)
    print("=" * 50)
    val_x, val_y = hl_dataset.get_val_X_Y(seq_len=SEQ_LEN, return_auxiliary_targets=True)
    val_dataset = HLMultiTaskDataset(val_x, val_y)

    input_dim = train_dataset[0][0].shape[-1]
    if hl_model is None:
        hl_model = HLActionStartModel(input_dim, 128) if mode in ("binary", "action_start", "start") else HLMultihead(input_dim, 128, 6)
    trainer = TrainTorchModel(train_dataset, val_dataset, model=hl_model)

    if mode in ["binary", "action_start", "start"]:
        train_info = trainer.train_action_start(model_name, action_name=action_name, checkpoint_meta={
            "preprocessing_version": 1,
            "features": list(hl_dataset.feature_list),
            "feature_transforms": FEATURE_TRANSFORMS,
            "feature_defaults": FEATURE_DEFAULTS,
            "seq_len": SEQ_LEN,
            "input_dim": input_dim,
            "output_dim": 1,
            "seed": seed,
        })
        print("Training Complete.")
        for info, data in train_info.items():
            print(f"  {info}: {data}")

        print(
              f"Config: \n"
              f"  model: \n{hl_model}\n"
              f"  loss_type: BCEWithLogitsLoss\n"
              f"  heal_loss_weight: {HEAL_LOSS_WEIGHT}\n"
              f"  pearl_loss_weight: {PEARL_LOSS_WEIGHT}\n"
              f"  seq_len: {SEQ_LEN}\n"
        )

        trainer.eval_action_start(hl_model, action_name=action_name, graph=True, confusion_matrix=True)

        return hl_model, trainer

    train_info = trainer.train(model_name, auxiliary_heads=auxiliary_heads, checkpoint_meta={
        "preprocessing_version": 1,
        "features": list(hl_dataset.feature_list),
        "feature_transforms": FEATURE_TRANSFORMS,
        "feature_defaults": FEATURE_DEFAULTS,
        "seq_len": SEQ_LEN,
        "input_dim": input_dim,
        "output_dim": 6,
        "seed": seed,
    })
    print("Training Complete.")
    for info, data in train_info.items():
        print(f"  {info}: {data}")

    print(f"Config: \n"
          f"  model: \n{hl_model}\n"
          # f"  train_distribution: {train_distribution}\n"
          f"  loss_type: CrossEntropy\n"
          f"  weights: {CLASS_WEIGHTS.numpy()}\n"
          f"  loss_weights: {LOSS_WEIGHTS}\n"
          f"  pot_urgency_beta: {POT_URGENCY_BETA}\n"
          f"  heal_loss_weight: {HEAL_LOSS_WEIGHT}\n"
          f"  pearl_loss_weight: {PEARL_LOSS_WEIGHT}\n"
          f"  seq_len: {SEQ_LEN}\n"
          )
    # print(
    #     f"Notes: \n"
    #     f"  added checkpoint storage based on heal/pearl initiation improvements [17]"
    #     f"  architecture test: MLP hidden encoder [18]"
    # )

    print("\nValidation: restored minimum-loss checkpoint")
    trainer.eval(hl_model, graph=True, confusion_matrix=True, inspect_initiation=True, auxiliary_heads=auxiliary_heads)

    return hl_model, trainer

def test_model(model_name="hl_multihead_?.pt", hl_model=None, test_loader=None):
    if hl_model is None:
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "models" / model_name
        hl_model = TrainTorchModel.load_model(model_path)

    metadata = getattr(hl_model, "checkpoint_meta", {})
    hl_dataset = HLDecisionDataset(use_cache=True, build_cache=False, features=metadata.get("features"))

    trainer = TrainTorchModel(model=hl_model)

    if test_loader is None:
        test_x, test_y = hl_dataset.get_test_X_Y(seq_len=metadata.get("seq_len", SEQ_LEN), return_auxiliary_targets=True)
        test_dataset = HLMultiTaskDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # print("\n" + "=" * 50)
    # print("Test: ")
    if isinstance(hl_model, HLActionStartModel):
        return trainer.eval_action_start(hl_model, action_name=metadata["action_name"], graph=True,
                                         confusion_matrix=True, dataloader=test_loader)

    trainer.eval(hl_model, graph=True, confusion_matrix=True, interpret_shap=False, inspect_initiation=True, dataloader=test_loader,
                 test_X=test_x if test_loader is None else None)

if __name__ == "__main__":
    # hl_model, trainer = train_eval_model(model_name="hl_multihead_21.pt", auxiliary_heads=["heal", "pearl", "pot_soon", "pot_urgency"])
    # test_model("hl_multihead_16_benchmark.pt")

    # heal_model = HLActionStartModel(input_dim=65, hidden_dim=128, output_dim=1)
    # pearl_model = HLActionStartModel(input_dim=65, hidden_dim=128, output_dim=1)
    # heal_model, trainer = train_eval_model(model_name="hl_heal_start_model_2.pt", hl_model=heal_model, mode="start", action_name="heal_start")
    # pearl_model, trainer = train_eval_model(
    #     model_name="hl_pearl_start_model_1.pt",
    #     hl_model=pearl_model,
    #     mode="start",
    #     action_name="pearl_start",
    # )

    # state_model = HLMultiheadNoAux(input_dim=65, hidden_dim=128, output_dim=6)
    # state_model, trainer = train_eval_model("hl_no_aux_1.pt", state_model, auxiliary_heads=None)

    ds = HLDecisionDataset(use_cache=True, build_cache=False)
    train_x, train_y = ds.get_train_X_Y(seq_len=SEQ_LEN, return_auxiliary_targets=True, msg=False)
    train_dataset = HLMultiTaskDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_x, val_y = ds.get_val_X_Y(seq_len=SEQ_LEN, return_auxiliary_targets=True)
    val_dataset = HLMultiTaskDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_prefix = "hl_multihead_21"
    for name in ("loss", "heal", "pearl"):
        model_name = f"{model_prefix}_{name}.pt"
        print("=" * 50)
        print("Val:", model_name)
        test_model(model_name=model_name, test_loader=val_loader)
        print("\n\n")
