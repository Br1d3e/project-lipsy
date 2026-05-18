# Python TCP server to transfer data between forge client and models

import os
import socket
import sys
import time
from collections import deque
from collections import OrderedDict

import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AIM_DIR = os.path.join(BASE_DIR, "aim")
STRAFE_DIR = os.path.join(BASE_DIR, "strafe")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, AIM_DIR)
sys.path.insert(0, STRAFE_DIR)

from aim.train_aim_model import Aim1dCNN, AimGRU, AimLSTM, AimSimpleMLP, AimStackedMLP
from aim.train_aim_model import SEQ_LEN as AIM_SEQ_LEN

from strafe.train_strafe_model import Strafe1dCNN, StrafeStackedMLP
from strafe.train_strafe_model import SEQ_LEN as STRAFE_SEQ_LEN


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OBS_SIZE = 17
AIM_FEATURE_SIZE = 16
STRAFE_FEATURE_SIZE = 16
AIM_ACTION_SIZE = 2
STRAFE_ACTION_SIZE = 1
ACTION_SIZE = AIM_ACTION_SIZE + 1 + STRAFE_ACTION_SIZE
STRAFE_OUTPUT_SIZE = 3

MODEL_CLASSES = {
    "AimSimpleMLP": AimSimpleMLP,
    "AimStackedMLP": AimStackedMLP,
    "AimLSTM": AimLSTM,
    "AimGRU": AimGRU,
    "Aim1dCNN": Aim1dCNN,
    "StrafeStackedMLP": StrafeStackedMLP,
    "Strafe1dCNN": Strafe1dCNN,
}


def checkpoint_model_input_dim(model_class_name, checkpoint, fallback_input_dim):
    if model_class_name.endswith("1dCNN"):
        conv_weight = checkpoint["model_state_dict"].get("layers.0.weight")
        if conv_weight is not None:
            return int(conv_weight.shape[1])
    return int(checkpoint.get("input_dim", fallback_input_dim))


def checkpoint_cnn_layout(checkpoint):
    state_dict = checkpoint["model_state_dict"]
    conv_channels = []
    linear_channels = []
    conv_indices = []
    linear_indices = []
    for idx in range(0, 32, 2):
        weight = state_dict.get("layers.%d.weight" % idx)
        if weight is None or len(weight.shape) != 3:
            break
        conv_channels.append(int(weight.shape[0]))
        conv_indices.append(idx)

    for idx in range(0, 32):
        weight = state_dict.get("layers.%d.weight" % idx)
        if weight is None or len(weight.shape) != 2:
            continue
        linear_indices.append(idx)
        out_features = int(weight.shape[0])
        if out_features != int(checkpoint.get("output_dim", 2)):
            linear_channels.append(out_features)

    relu_after_final_conv = False
    if conv_indices and linear_indices:
        relu_after_final_conv = linear_indices[0] == conv_indices[-1] + 4
    return conv_channels, linear_channels, relu_after_final_conv


def load_model(path: str, input_dim: int, output_dim: int, default_model_class_name="Aim1dCNN",
               expected_model_class_name=None):
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name, model_class in MODEL_CLASSES.items():
            setattr(main_module, name, model_class)

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        model_class_name = loaded.get("model_class", default_model_class_name)
        if expected_model_class_name is not None and model_class_name != expected_model_class_name:
            raise ValueError(
                "Checkpoint %s has model_class=%s, expected %s. "
                "Retrain or point MODEL_PATH at the correct checkpoint."
                % (path, model_class_name, expected_model_class_name)
            )
        model_class = MODEL_CLASSES[model_class_name]
        kwargs = {
            "input_dim": checkpoint_model_input_dim(model_class_name, loaded, input_dim),
            "output_dim": loaded.get("output_dim", output_dim),
        }
        if model_class_name == "Aim1dCNN":
            conv_channels, linear_channels, relu_after_final_conv = checkpoint_cnn_layout(loaded)
            kwargs["conv_channels"] = conv_channels
            kwargs["linear_channels"] = linear_channels
            kwargs["relu_after_final_conv"] = relu_after_final_conv
        model = model_class(**kwargs)
        model.load_state_dict(loaded["model_state_dict"])
        model.checkpoint_meta = loaded
        return model
    if isinstance(loaded, OrderedDict):
        model_class = MODEL_CLASSES[default_model_class_name]
        model = model_class(input_dim=input_dim, output_dim=output_dim)
        model.load_state_dict(loaded)
        return model
    return loaded

AIM_MODEL_CONFIG = {
    "MODEL_PATH": os.path.join(AIM_DIR, "aim_cnn_8.pth"),
    "INPUT_DIM" : AIM_FEATURE_SIZE * AIM_SEQ_LEN,
    "OUTPUT_DIM": AIM_ACTION_SIZE
}

STRAFE_MODEL_CONFIG = {
    "MODEL_PATH": os.path.join(STRAFE_DIR, "strafe_cnn_3.pth"),
    "INPUT_DIM": STRAFE_FEATURE_SIZE * STRAFE_SEQ_LEN,
    "OUTPUT_DIM": STRAFE_OUTPUT_SIZE
}

host = "127.0.0.1"
port = 33333

def recv_exact(conn: socket.socket, size: int):
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Client connection failed.")
        data += chunk
    return data

def read_obs_packet(conn: socket.socket):
    BODY_LENGTH = 4 + 8 + 4 * OBS_SIZE  # int32 seq + int64 nanotime + float32 observation vector

    # Header: body length
    header = recv_exact(conn, 4)
    body_len = int.from_bytes(header, byteorder="little", signed=False)

    if body_len != BODY_LENGTH:
        got_obs_size = (body_len - 12) // 4 if body_len >= 12 and (body_len - 12) % 4 == 0 else None
        raise ValueError(
            "Body length mismatch: expected %d bytes (%d obs floats), got %d bytes (%s obs floats)"
            % (BODY_LENGTH, OBS_SIZE, body_len, got_obs_size)
        )

    # Payload: int32 seq + int64 nanotime + float32 observation vector
    payload = recv_exact(conn, body_len)
    seq = int.from_bytes(payload[:4], byteorder="little", signed=False)
    nanotime = int.from_bytes(payload[4:12], byteorder="little", signed=False)
    obs_bytes = payload[12:]
    obs = np.frombuffer(obs_bytes, dtype="<f4")
    return [seq, nanotime, obs]


def get_aim_data(data: np.ndarray):
    if len(data) < AIM_FEATURE_SIZE:
        raise ValueError("Aim observation too short: expected %d floats, got %d" % (AIM_FEATURE_SIZE, len(data)))
    return data[:AIM_FEATURE_SIZE]

def get_strafe_data(data: np.ndarray):
    if len(data) < OBS_SIZE:
        raise ValueError("Strafe observation too short: expected %d, got %d" % (OBS_SIZE, len(data)))
    # Java obs layout:
    # 0..3 motion/yaw, 4 myPitch, 5..15 movement/target/aim diffs, 16 myDeltaYaw.
    # Strafe model FEATURES are:
    # myMotionX/Y/Z, myYaw, myDeltaYaw, myMoveForward, myMoveStrafing,
    # targetRelX/Y/Z, targetDistance, targetMotionX/Y/Z, yawDiffToTarget, pitchDiffToTarget.
    return np.asarray([
        data[0], data[1], data[2], data[3], data[16],
        data[5], data[6], data[7], data[8], data[9], data[10],
        data[11], data[12], data[13], data[14], data[15],
    ], dtype="<f4")

def build_action_packet(seq: int, actions: list):
    actions = np.asarray(actions, dtype="<f4").reshape(-1)
    action_size = len(actions)  # Action vector size
    BODY_LENGTH = 4 + 8 + 4 * action_size  # int32 seq + int64 nanotime + float32 action vector

    packet = bytearray()
    # Header: body length
    packet += BODY_LENGTH.to_bytes(4, byteorder="little", signed=False)
    # Payload: int32 seq + int64 nanotime + float32 action vector
    packet += seq.to_bytes(4, byteorder="little", signed=False)
    nanotime = time.time_ns()
    packet += nanotime.to_bytes(8, byteorder="little", signed=False)
    packet += actions.tobytes()

    return packet


def infer_flattened_input_dim(model, fallback_input_dim: int):
    meta = getattr(model, "checkpoint_meta", None)
    if isinstance(meta, dict) and meta.get("input_dim") is not None:
        return int(meta["input_dim"])

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            return int(module.in_features)
    return fallback_input_dim


def infer_seq_len(model, feature_size: int, fallback_input_dim: int):
    meta = getattr(model, "checkpoint_meta", None)

    if isinstance(meta, dict):
        if meta.get("seq_len") is not None:
            return int(meta["seq_len"])

        if meta.get("flat_input_dim") is not None:
            flat_input_dim = int(meta["flat_input_dim"])
            if flat_input_dim % feature_size != 0:
                raise ValueError(
                    f"flat_input_dim {flat_input_dim} not divisible by feature_size {feature_size}"
                )
            return flat_input_dim // feature_size

        if meta.get("input_dim") is not None:
            input_dim = int(meta["input_dim"])
            if input_dim % feature_size != 0:
                raise ValueError(
                    f"input_dim {input_dim} not divisible by feature_size {feature_size}"
                )
            return input_dim // feature_size

    return fallback_input_dim // feature_size


def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"Server running at {host}/{port}.")

    aim_model = load_model(
        AIM_MODEL_CONFIG["MODEL_PATH"],
        AIM_MODEL_CONFIG["INPUT_DIM"],
        AIM_MODEL_CONFIG["OUTPUT_DIM"],
        default_model_class_name="Aim1dCNN",
    )
    aim_model.to(device)
    aim_model.eval()
    aim_seq_len = infer_seq_len(aim_model, AIM_FEATURE_SIZE, AIM_MODEL_CONFIG["INPUT_DIM"])

    strafe_model = load_model(
        STRAFE_MODEL_CONFIG["MODEL_PATH"],
        STRAFE_MODEL_CONFIG["INPUT_DIM"],
        STRAFE_MODEL_CONFIG["OUTPUT_DIM"],
        default_model_class_name="Strafe1dCNN",
        expected_model_class_name="Strafe1dCNN",
    )
    strafe_model.to(device)
    strafe_model.eval()
    strafe_seq_len = infer_seq_len(strafe_model, STRAFE_FEATURE_SIZE, STRAFE_MODEL_CONFIG["INPUT_DIM"])
    print("Loaded models: aim_seq_len=%d strafe_seq_len=%d" % (aim_seq_len, strafe_seq_len))

    while True:
        conn, addr = server.accept()
        print("Client connected", addr)
        aim_history = deque(maxlen=aim_seq_len)

        strafe_history = deque(maxlen=strafe_seq_len)

        try:
            while True:
                seq, nanotime, obs = read_obs_packet(conn)

                # Aiming
                aim_x = get_aim_data(obs)
                aim_history.append(aim_x)
                while len(aim_history) < aim_seq_len:
                    aim_history.appendleft(aim_x)

                aim_tensor = torch.tensor(np.array(aim_history), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    aim_y = aim_model(aim_tensor).reshape(-1).cpu().numpy()[:2]

                # Strafe
                strafe_x = get_strafe_data(obs)
                strafe_history.append(strafe_x)
                while len(strafe_history) < strafe_seq_len:
                    strafe_history.appendleft(strafe_x)

                strafe_tensor = torch.tensor(np.array(strafe_history), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    strafe_logits = strafe_model(strafe_tensor).reshape(-1).cpu().numpy()
                    strafe_y = float(np.argmax(strafe_logits) - 1)

                exp_logits = np.exp(strafe_logits - np.max(strafe_logits))
                probs = exp_logits / exp_logits.sum()
                pred_cls = int(np.argmax(probs))
                strafe_confidence = float(probs[pred_cls])

                
                # Keep legacy Java action layout:
                # [deltaYaw, deltaPitch, strafeAxis, strafeConf]
                action_list = np.concatenate((aim_y, np.asarray([strafe_y, strafe_confidence], dtype="<f4"))).tolist()
                packet = build_action_packet(seq, action_list)
                conn.sendall(packet)
        except (ConnectionError, ValueError) as e:
            print("Client disconnected:", e)
        finally:
            conn.close()

if __name__ == "__main__":
    run_server()
