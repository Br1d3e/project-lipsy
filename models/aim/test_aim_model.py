import torch
import matplotlib.pyplot as plt
import numpy as np
from train_aim_model import AimDataset, AimLSTM

# ===== Configuration =====
FOLDER = "/Users/br1d3e/Library/CloudStorage/OneDrive-Personal/Project Lipsy - Continued/test/recordings/aim"
# FOLDER = "D:/OneDrive/Project Lipsy - Continued/test/recordings/aim"
MODEL_PATH = "aim_model_unscaled.pth"
SEQ_LEN = 20
BATCH_SIZE = 32

# ===== Load dataset & model =====
dataset = AimDataset(FOLDER, seq_len=SEQ_LEN)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("mps")
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.to(device)
model.eval()

# ===== Collect predictions =====
y_true_all, y_pred_all = [], []

with torch.no_grad():
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb)
        y_pred_all.append(pred.cpu().numpy())
        y_true_all.append(yb.numpy())

y_true = np.concatenate(y_true_all, axis=0)
y_pred = np.concatenate(y_pred_all, axis=0)

# ===== Inverse scale =====
# y_true_inv = dataset.scaler_y.inverse_transform(y_true)
y_true_inv = y_true
# y_pred_inv = dataset.scaler_y.inverse_transform(y_pred)
y_pred_inv = y_pred

# ===== Separate columns =====
yaw_true, pitch_true = y_true_inv[:, 0], y_true_inv[:, 1]
yaw_pred, pitch_pred = y_pred_inv[:, 0], y_pred_inv[:, 1]

# ===== Scatter plots =====
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(yaw_true, yaw_pred, s=5, alpha=0.5)
plt.xlabel("True yawDiff")
plt.ylabel("Pred yawDiff")
plt.title("YawDiff Prediction")
plt.plot([yaw_true.min(), yaw_true.max()],
         [yaw_true.min(), yaw_true.max()], 'r--')

plt.subplot(1, 2, 2)
plt.scatter(pitch_true, pitch_pred, s=5, alpha=0.5)
plt.xlabel("True pitchDiff")
plt.ylabel("Pred pitchDiff")
plt.title("PitchDiff Prediction")
plt.plot([pitch_true.min(), pitch_true.max()],
         [pitch_true.min(), pitch_true.max()], 'r--')

plt.tight_layout()
plt.show()

# ===== Time series comparison =====
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(yaw_true[:300], label='True')
plt.plot(yaw_pred[:300], label='Pred', alpha=0.7)
plt.title("YawDiff over time")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(pitch_true[:300], label='True')
plt.plot(pitch_pred[:300], label='Pred', alpha=0.7)
plt.title("PitchDiff over time")
plt.legend()

plt.tight_layout()
plt.show()
