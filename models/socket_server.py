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
WTAP_DIR = os.path.join(BASE_DIR, "wtap")
BLOCKHIT_DIR = os.path.join(BASE_DIR, "blockhit")
HL_DIR = os.path.join(BASE_DIR, "hl_decision")
HEAL_TRIGGER_DIR = os.path.join(HL_DIR, "heal_trigger")
PEARL_TRIGGER_DIR = os.path.join(HL_DIR, "pearl_trigger")
HL_MODEL_DIR = os.path.join(HL_DIR, "models")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, AIM_DIR)
sys.path.insert(0, STRAFE_DIR)
sys.path.insert(0, WTAP_DIR)
sys.path.insert(0, BLOCKHIT_DIR)
sys.path.insert(0, HEAL_TRIGGER_DIR)
sys.path.insert(0, PEARL_TRIGGER_DIR)
sys.path.insert(0, HL_MODEL_DIR)

from aim.train_aim_model import Aim1dCNN, AimGRU, AimLSTM, AimSimpleMLP, AimStackedMLP
from aim.train_aim_model import SEQ_LEN as AIM_SEQ_LEN

from strafe.train_strafe_model import Strafe1dCNN, StrafeStackedMLP
from strafe.train_strafe_model import SEQ_LEN as STRAFE_SEQ_LEN

from wtap.train_wtap_model import Wtap1dCNN
from wtap.train_wtap_model import SEQ_LEN as WTAP_SEQ_LEN
from wtap.wtap_datasets import FEATURES as WTAP_FEATURES

from blockhit.train_blockhit_model import BlockhitStackedMLP, Blockhit1dCNN
from blockhit.train_blockhit_model import SEQ_LEN as BLOCKHIT_SEQ_LEN

from hl_decision.heal_trigger.train_heal_trigger_model import MODEL_PATH as HEAL_TRIGGER_MODEL_PATH
if not os.path.exists(HEAL_TRIGGER_MODEL_PATH):
    HEAL_TRIGGER_MODEL_PATH = os.path.join(HEAL_TRIGGER_DIR, "heal_trigger_xgb_5.json")
from hl_decision.pearl_trigger.train_pearl_trigger_model import MODEL_PATH as PEARL_TRIGGER_MODEL_PATH
from hl_decision.pearl_trigger.train_pearl_trigger_model import PearlTriggerLSTM, SEQ_LEN as PEARL_SEQ_LEN

if not os.path.exists(PEARL_TRIGGER_MODEL_PATH):
    PEARL_TRIGGER_MODEL_PATH = os.path.join(PEARL_TRIGGER_DIR, "pearl_trigger_xgb_4.json")

from hl_decision.train_hl_decision_model import HLMultihead, FEATURE_TRANSFORMS, SEQ_LEN as HL_SEQ_LEN
HL_MODEL_PATH = os.path.join(HL_MODEL_DIR, "hl_multihead_21_heal.pt")

import xgboost as xgb

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
pearl_lstm_device = torch.device("cpu")
OBS_FEATURES = [
    "myMotionX",
    "myMotionY",
    "myMotionZ",
    "myYaw",
    "myPitch",
    "myDeltaYaw",
    "myDeltaPitch",
    "myHp",
    "myHurtTime",
    "myMoveForward",
    "myMoveStrafing",
    "mySwingProgress",
    "myUsingItem",
    "myHoldingSword",
    "myBlockingSword",
    "myHasSpeedPotion",
    "mySpeedDurationTicks",
    "myFoodLevel",
    "myCombo",
    "targetCombo",
    "pearlCooldownTicksRemaining",
    "enemyPearlAvailable",
    "enemyPearlCooldownTicksRemaining",
    "hotbarHealthPotions",
    "inventoryFood",
    "inventorySpeedPotions",
    "inventoryHealthPotions",
    "inventoryPearls",
    "crosshairOnTarget",
    "ticksSinceLastAttack",
    "ticksSinceLastBlock",
    "ticksSinceLastPot",
    "ticksSinceLastEat",
    "ticksSinceLastSpeed",
    "ticksSinceLastDamageTaken",
    "ticksSinceLastHitLanded",
    "ticksSinceLastPearl",
    "ticksSinceLastEnemyPearl",
    "blockDistanceFront",
    "blockDistanceBack",
    "blockDistanceLeft",
    "blockDistanceRight",
    "aStrafe",
    "aAttack",
    "aUseItem",
    "targetExists",
    "targetRelX",
    "targetRelY",
    "targetRelZ",
    "targetDistance",
    "targetMotionX",
    "targetMotionY",
    "targetMotionZ",
    "targetYaw",
    "targetPitch",
    "targetDeltaYaw",
    "targetDeltaPitch",
    "targetEstimatedHp",
    "targetEstimatedPotions",
    "targetEstimatedPearls",
    "targetHurtTime",
    "targetSwingProgress",
    "targetUsingItem",
    "targetHasSpeedPotion",
    "targetSpeedDurationTicks",
    "targetFood",
    "targetEating",
    "targetSpeeding",
    "targetPearling",
    "targetPotting",
    "yawDiffToTarget",
    "pitchDiffToTarget",
    "hpDiff",
    "canHeal",
    "canPearl",
    "canSpeed",
    "canEat",
    "s_attack",
    "s_heal",
    "s_pearl",
    "s_speed",
    "s_eat",
    "s_refill",
    'myHitPressure',
    'targetHitPressure',
    'hitsLandedLastK',
    'hitsTakenLastK',
    'netHitsLastK',
    'ticksSinceCombat',
    'ticksInCurrentState',
    'targetRadialVelocity',
    'targetOutwardVelocity',
    'targetFacingMeCos',
    'distanceDelta5',
    'distanceDelta20',
    'hitsBeforeTargetHeals',
    'targetHoldingFood',
    'targetHoldingSpeedPotion',
    'targetHoldingHealPotion',
    'ping',
]
OBS_INDEX = {name: i for i, name in enumerate(OBS_FEATURES)}
OBS_SIZE = len(OBS_FEATURES)
AIM_FEATURE_SIZE = 18
STRAFE_FEATURE_SIZE = 16
WTAP_FEATURE_SIZE = len(WTAP_FEATURES)
BLOCKHIT_FEATURE_SIZE = 14
HL_DECISION_FEATURES = [
    "myHp",
    "myFoodLevel",
    "myHasSpeedPotion",
    "mySpeedDurationTicks",
    "myCombo",
    "targetCombo",
    "myHitPressure",
    "targetHitPressure",
    "pearlCooldownTicksRemaining",
    "enemyPearlAvailable",
    "enemyPearlCooldownTicksRemaining",
    "hotbarHealthPotions",
    "inventoryFood",
    "inventorySpeedPotions",
    "inventoryHealthPotions",
    "inventoryPearls",
    "ticksSinceLastAttack",
    "ticksSinceLastBlock",
    "ticksSinceLastPot",
    "ticksSinceLastEat",
    "ticksSinceLastSpeed",
    "ticksSinceLastDamageTaken",
    "ticksSinceLastHitLanded",
    "ticksSinceLastPearl",
    "ticksSinceLastEnemyPearl",
    "hitsLandedLastK",
    "hitsTakenLastK",
    "netHitsLastK",
    "ticksSinceCombat",
    # "ticksInCurrentState",
    "blockDistanceFront",
    "blockDistanceBack",
    "blockDistanceLeft",
    "blockDistanceRight",
    "targetExists",
    "targetDistance",
    'targetRelX',
    'targetRelZ',
    "targetMotionX",
    "targetMotionZ",
    'targetYaw',
    'targetPitch',
    'targetDeltaYaw',
    'targetDeltaPitch',
    "targetRadialVelocity",
    "targetOutwardVelocity",
    "targetFacingMeCos",
    "distanceDelta5",
    "distanceDelta20",
    "targetEstimatedHp",
    "targetEstimatedPotions",
    "hitsBeforeTargetHeals",
    "targetHasSpeedPotion",
    "targetSpeedDurationTicks",
    "targetFood",
    "targetHoldingFood",
    "targetHoldingSpeedPotion",
    "targetHoldingHealPotion",
    "targetPearling",
    "targetPotting",
    "hpDiff",
    "canHeal",
    "canPearl",
    "canSpeed",
    "canEat",
    # "s_attack",
    # "s_heal",
    # "s_pearl",
    # "s_speed",
    # "s_eat",
    # "s_refill",
    "ping",
]
HL_DECISION_FEATURE_SIZE = len(HL_DECISION_FEATURES)
AIM_ACTION_SIZE = 2
STRAFE_ACTION_SIZE = 1
WTAP_ACTION_SIZE = 1
BLOCKHIT_ACTION_SIZE = 1
HEAL_TRIGGER_ACTION_SIZE = 1
PEARL_TRIGGER_ACTION_SIZE = 2
ACTION_SIZE = AIM_ACTION_SIZE + 1 + STRAFE_ACTION_SIZE + BLOCKHIT_ACTION_SIZE + HEAL_TRIGGER_ACTION_SIZE + PEARL_TRIGGER_ACTION_SIZE + WTAP_ACTION_SIZE
STRAFE_OUTPUT_SIZE = 3
BLOCKHIT_OUTPUT_SIZE = 2
HL_DECISION_OUTPUT_SIZE = 6

MODEL_CLASSES = {
    "AimSimpleMLP": AimSimpleMLP,
    "AimStackedMLP": AimStackedMLP,
    "AimLSTM": AimLSTM,
    "AimGRU": AimGRU,
    "Aim1dCNN": Aim1dCNN,
    "StrafeStackedMLP": StrafeStackedMLP,
    "Strafe1dCNN": Strafe1dCNN,
    "Wtap1dCNN": Wtap1dCNN,
    "BlockhitStackedMLP": BlockhitStackedMLP,
    "Blockhit1dCNN": Blockhit1dCNN,
    "PearlTriggerLSTM": PearlTriggerLSTM,
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


def checkpoint_mlp_hidden_channels(checkpoint):
    state_dict = checkpoint["model_state_dict"]
    output_dim = int(checkpoint.get("output_dim", 2))
    hidden_channels = []
    for idx in range(0, 64):
        weight = state_dict.get("layers.%d.weight" % idx)
        if weight is None or len(weight.shape) != 2:
            continue
        out_features = int(weight.shape[0])
        if out_features != output_dim:
            hidden_channels.append(out_features)
    return hidden_channels


def checkpoint_lstm_layout(checkpoint):
    state_dict = checkpoint["model_state_dict"]
    weight_hh_l0 = state_dict.get("lstm.weight_hh_l0")
    hidden_dim = int(checkpoint.get("hidden_dim", weight_hh_l0.shape[1] if weight_hh_l0 is not None else 128))
    layer_ids = []
    for key in state_dict:
        if key.startswith("lstm.weight_ih_l"):
            try:
                layer_ids.append(int(key.rsplit("_l", 1)[1]))
            except (IndexError, ValueError):
                pass
    num_layers = int(checkpoint.get("num_layers", max(layer_ids) + 1 if layer_ids else 1))
    return hidden_dim, num_layers


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
        if model_class_name == "BlockhitStackedMLP":
            kwargs["hidden_channels"] = checkpoint_mlp_hidden_channels(loaded)
        if model_class_name == "PearlTriggerLSTM":
            hidden_dim, num_layers = checkpoint_lstm_layout(loaded)
            kwargs["hidden_dim"] = hidden_dim
            kwargs["num_layers"] = num_layers
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

def load_hl_model(path, device=None, return_transforms=False):
    if device is None:
        device = device

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    feature_transforms = None
    feature_defaults = None
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        feature_transforms = checkpoint.get("feature_transforms", None)
        feature_defaults = checkpoint.get("feature_defaults", None)
        model = HLMultihead.from_checkpoint(checkpoint)
    else:
        model = checkpoint
    model.checkpoint_meta = checkpoint
    model.to(device)
    model.eval()

    if feature_transforms is None and return_transforms:
        raise ValueError("feature_transforms not available")

    if return_transforms:
        if feature_defaults is None:
            raise ValueError("feature_defaults not available")
        return model, feature_transforms, feature_defaults

    return model

AIM_MODEL_CONFIG = {
    "MODEL_PATH": os.path.join(AIM_DIR, "aim_lstm_9.pth"),
    "INPUT_DIM" : AIM_FEATURE_SIZE,
    "OUTPUT_DIM": AIM_ACTION_SIZE
}

STRAFE_MODEL_CONFIG = {
    "MODEL_PATH": os.path.join(STRAFE_DIR, "strafe_cnn_3.pth"),
    "INPUT_DIM": STRAFE_FEATURE_SIZE * STRAFE_SEQ_LEN,
    "OUTPUT_DIM": STRAFE_OUTPUT_SIZE
}

WTAP_MODEL_CONFIG = {
    "MODEL_PATH": os.path.join(WTAP_DIR, "wtap_cnn_3.pth"),
    "INPUT_DIM": WTAP_FEATURE_SIZE * WTAP_SEQ_LEN,
    "OUTPUT_DIM": 3,
}

BLOCKHIT_MODEL_CONFIG = {
    "MODEL_PATH": os.path.join(BLOCKHIT_DIR, "blockhit_stacked_mlp_9.pth"),
    "INPUT_DIM": BLOCKHIT_FEATURE_SIZE * BLOCKHIT_SEQ_LEN,
    "OUTPUT_DIM": BLOCKHIT_OUTPUT_SIZE
}

PEARL_TRIGGER_MODEL_CONFIG = {
    "MODEL_PATH": os.path.join(PEARL_TRIGGER_DIR, "pearl_trigger_lstm_3.pth"),
    "INPUT_DIM": HL_DECISION_FEATURE_SIZE,
    "OUTPUT_DIM": PEARL_TRIGGER_ACTION_SIZE
}

HL_MODEL_CONFIG = {
    "MODEL_PATH": HL_MODEL_PATH,
    "DEFAULT_FEATURE_TRANSFORMS": FEATURE_TRANSFORMS,
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
    return get_features(data, [
        "myMotionX", "myMotionY", "myMotionZ", "myYaw", "myPitch", "myDeltaYaw", "myDeltaPitch",
        "myMoveForward", "myMoveStrafing", "targetRelX", "targetRelY", "targetRelZ", "targetDistance",
        "targetMotionX", "targetMotionY", "targetMotionZ", "yawDiffToTarget", "pitchDiffToTarget",
    ])

def get_strafe_data(data: np.ndarray):
    return get_features(data, [
        "myMotionX", "myMotionY", "myMotionZ", "myYaw", "myDeltaYaw", "myMoveForward", "myMoveStrafing",
        "targetRelX", "targetRelY", "targetRelZ", "targetDistance",
        "targetMotionX", "targetMotionY", "targetMotionZ", "yawDiffToTarget", "pitchDiffToTarget",
    ])

def get_wtap_data(data: np.ndarray):
    return get_features(data, WTAP_FEATURES)


def get_blockhit_data(data: np.ndarray):
    return get_features(data, [
        "aAttack", "aStrafe", "targetHurtTime", "myHurtTime", "ticksSinceLastDamageTaken", "ticksSinceLastAttack",
        "targetDistance", "targetSwingProgress", "mySwingProgress", "myHoldingSword", "myUsingItem",
        "crosshairOnTarget", "yawDiffToTarget", "pitchDiffToTarget",
    ])

def get_hl_decision_data(data: np.ndarray, transforms: dict, defaults: dict):
    """Match dataset preprocessing: normalize first, then fill checkpoint defaults."""
    if len(data) < OBS_SIZE:
        raise ValueError("Observation too short: expected %d floats, got %d" % (OBS_SIZE, len(data)))
    hl_data = []
    for name in HL_DECISION_FEATURES:
        value = data[OBS_INDEX[name]]
        if name in transforms:
            value = apply_hl_transform(value, transforms[name])
        if np.isnan(value):
            value = defaults.get(name, np.nan)
        if not np.isfinite(value):
            raise ValueError(f"Non-finite HL feature after preprocessing: {name}")
        hl_data.append(value)
    return np.asarray(hl_data, dtype="<f4")

def apply_hl_transform(raw_data, transform: dict):
    """
    Applies HL transforms to scaler data.
    :param raw_data: Scaler data as float
    :param transform: transform info of a unique feature field
    :return: transformed data
    """
    expected_keys = ["type", "cap", "divisor"]
    if any(key not in list(transform.keys()) for key in expected_keys):
        raise ValueError(f"Invalid HL transform: {transform}")

    type = transform["type"]
    if type != "log1p":
        raise NotImplementedError(f"HL transform type {type} is not supported")
    cap = transform["cap"]
    divisor = transform["divisor"]
    # The dataset masks negative raw timers before log1p; defaults are applied later.
    if np.isnan(raw_data) or raw_data < 0:
        return np.nan
    return np.log1p(min(raw_data, cap)) / divisor

def get_features(data: np.ndarray, feature_names):
    if len(data) < OBS_SIZE:
        raise ValueError("Observation too short: expected %d floats, got %d" % (OBS_SIZE, len(data)))
    try:
        return np.asarray([data[OBS_INDEX[name]] for name in feature_names], dtype="<f4")
    except KeyError as exc:
        raise KeyError("Unknown observation feature: %s" % exc)

def logits_to_blockhit_action(logits, p_threshold=0.8):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    return 1.0 if probs[1] > p_threshold else 0.0

def logits_to_wtap_action(logits):
    return float(np.argmax(np.asarray(logits).reshape(-1)) - 1)

def probs_to_heal_trigger(probs, p_threshold=0.7):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    return 1.0 if probs[1] > p_threshold else 0.0

def logits_to_pearl_probs(logits):
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    return probs

def probs_to_pearl_trigger(probs, p_threshold=0.6):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    return 1.0 if probs[1] > p_threshold else 0.0

def multi_logits_to_probs(logits):
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    probs = torch.softmax(logits, dim=-1)
    return np.asarray(probs, dtype=float)

def multi_logits_to_class(logits):
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    return torch.argmax(logits, dim=-1).long().numpy()

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

def infer_hl_seq_len(model):
    meta = getattr(model, "checkpoint_meta", None)
    if isinstance(meta, dict) and meta.get("seq_len") is not None:
        return int(meta["seq_len"])
    return HL_SEQ_LEN

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

    wtap_model = load_model(
        WTAP_MODEL_CONFIG["MODEL_PATH"],
        WTAP_MODEL_CONFIG["INPUT_DIM"],
        WTAP_MODEL_CONFIG["OUTPUT_DIM"],
        default_model_class_name="Wtap1dCNN",
        expected_model_class_name="Wtap1dCNN",
    )
    wtap_model.to(device)
    wtap_model.eval()
    wtap_seq_len = infer_seq_len(wtap_model, WTAP_FEATURE_SIZE, WTAP_MODEL_CONFIG["INPUT_DIM"])

    blockhit_model = load_model(
        BLOCKHIT_MODEL_CONFIG["MODEL_PATH"],
        BLOCKHIT_MODEL_CONFIG["INPUT_DIM"],
        BLOCKHIT_MODEL_CONFIG["OUTPUT_DIM"],
        default_model_class_name="BlockhitStackedMLP",
        expected_model_class_name="BlockhitStackedMLP",
    )
    blockhit_model.to(device)
    blockhit_model.eval()
    blockhit_seq_len = infer_seq_len(blockhit_model, BLOCKHIT_FEATURE_SIZE, BLOCKHIT_MODEL_CONFIG["INPUT_DIM"])

    # heal_trigger_model = xgb.XGBClassifier()
    # heal_trigger_model.load_model(HEAL_TRIGGER_MODEL_PATH)
    # pearl_trigger_xgb = xgb.XGBClassifier()
    # pearl_trigger_xgb.load_model(PEARL_TRIGGER_MODEL_PATH)
    #
    # pearl_trigger_lstm = load_model(
    #     PEARL_TRIGGER_MODEL_CONFIG["MODEL_PATH"],
    #     PEARL_TRIGGER_MODEL_CONFIG["INPUT_DIM"],
    #     PEARL_TRIGGER_MODEL_CONFIG["OUTPUT_DIM"],
    #     default_model_class_name="PearlTriggerLSTM",
    #     expected_model_class_name="PearlTriggerLSTM",
    # )
    # pearl_trigger_lstm.to(pearl_lstm_device)
    # pearl_trigger_lstm.eval()
    #
    # pearl_seq_len =  infer_seq_len(pearl_trigger_lstm, HL_DECISION_FEATURE_SIZE, PEARL_TRIGGER_MODEL_CONFIG["INPUT_DIM"])

    hl_model, feature_transforms, feature_defaults = load_hl_model(
        HL_MODEL_CONFIG["MODEL_PATH"],
        device,
        True
    )
    hl_seq_len = infer_hl_seq_len(hl_model)

    print("Loaded models: aim_seq_len=%d strafe_seq_len=%d wtap_seq_len=%d blockhit_seq_len=%d hl_seq_len=%d"
          % (aim_seq_len, strafe_seq_len, wtap_seq_len, blockhit_seq_len, hl_seq_len))

    while True:
        conn, addr = server.accept()
        print("Client connected", addr)
        aim_history = deque(maxlen=aim_seq_len)
        strafe_history = deque(maxlen=strafe_seq_len)
        wtap_history = deque(maxlen=wtap_seq_len)
        blockhit_history = deque(maxlen=blockhit_seq_len)
        # pearl_trigger_history = deque(maxlen=pearl_seq_len)
        hl_history = deque(maxlen=hl_seq_len)

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

                # W-tap
                wtap_x = get_wtap_data(obs)
                wtap_history.append(wtap_x)
                while len(wtap_history) < wtap_seq_len:
                    wtap_history.appendleft(wtap_x)

                wtap_tensor = torch.tensor(np.array(wtap_history), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    wtap_logits = wtap_model(wtap_tensor).reshape(-1).cpu().numpy()
                    wtap_y = logits_to_wtap_action(wtap_logits)

                # Blockhit
                blockhit_x = get_blockhit_data(obs)
                blockhit_history.append(blockhit_x)
                while len(blockhit_history) < blockhit_seq_len:
                    blockhit_history.appendleft(blockhit_x)

                blockhit_tensor = torch.tensor(np.array(blockhit_history), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    blockhit_logits = blockhit_model(blockhit_tensor).reshape(-1).cpu().numpy()
                    blockhit_y = logits_to_blockhit_action(blockhit_logits, p_threshold=0.3)

                # High-level decision
                hl_decision_x = get_hl_decision_data(obs, feature_transforms, feature_defaults)
                hl_history.append(hl_decision_x)
                while len(hl_history) < hl_seq_len:
                    hl_history.appendleft(hl_decision_x)

                # # Heal trigger
                # heal_probs = heal_trigger_model.predict_proba(hl_decision_x.reshape(1, -1))[0]
                # heal_y = probs_to_heal_trigger(heal_probs)
                #
                # # Pearl trigger
                # pearl_probs = pearl_trigger_xgb.predict_proba(hl_decision_x.reshape(1, -1))[0]
                # pearl_prob = float(pearl_probs[1])
                # pearl_y = probs_to_pearl_trigger(pearl_probs, p_threshold=0.5)
                # pearl_tensor = torch.tensor(np.array(pearl_trigger_history), dtype=torch.float32).unsqueeze(0).to(pearl_lstm_device)
                # with torch.no_grad():
                #     pearl_logits = pearl_trigger_lstm(pearl_tensor).reshape(-1).cpu().numpy()
                # pearl_probs = logits_to_pearl_probs(pearl_logits)
                # pearl_prob = float(pearl_probs[1])
                # pearl_y = probs_to_heal_trigger(pearl_probs, p_threshold=0.8)

                hl_tensor = torch.tensor(np.array(hl_history), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    hl_pred = hl_model(hl_tensor)
                    hl_state_logits = hl_pred["state_logits"].reshape(-1).cpu().numpy()
                    hl_heal_logit = hl_pred["heal_logits"].reshape(-1).cpu()
                    hl_pearl_logit = hl_pred["pearl_logits"].reshape(-1).cpu()
                    hl_probs = multi_logits_to_probs(hl_state_logits)
                    hl_heal_prob = torch.sigmoid(hl_heal_logit).numpy()
                    hl_pearl_prob = torch.sigmoid(hl_pearl_logit).numpy()
                    hl_y = multi_logits_to_class(hl_state_logits)

                # [deltaYaw, deltaPitch, strafeAxis, strafeConf, blockhitUse, wtapAxis, hlState, hlProbs]
                action_list = np.concatenate((
                    aim_y,
                    np.asarray([strafe_y, strafe_confidence, blockhit_y, wtap_y, hl_y], dtype="<f4"),
                    hl_probs,
                    hl_heal_prob,
                    hl_pearl_prob
                )).tolist()
                packet = build_action_packet(seq, action_list)
                conn.sendall(packet)
        except (ConnectionError, ValueError) as e:
            print("Client disconnected:", e)
        finally:
            conn.close()

if __name__ == "__main__":
    run_server()
