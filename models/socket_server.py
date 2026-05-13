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
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, AIM_DIR)

from aim.train_aim_model import AimGRU, AimLSTM, AimSimpleMLP, AimStackedMLP, SEQ_LEN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODEL_CLASSES = {
    "AimSimpleMLP": AimSimpleMLP,
    "AimStackedMLP": AimStackedMLP,
    "AimLSTM": AimLSTM,
    "AimGRU": AimGRU,
}


def load_model(path: str, input_dim: int, output_dim: int):
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for name, model_class in MODEL_CLASSES.items():
            setattr(main_module, name, model_class)

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        model_class = MODEL_CLASSES.get(loaded.get("model_class", "AimStackedMLP"), AimStackedMLP)
        model = model_class(
            input_dim=loaded.get("input_dim", input_dim),
            output_dim=loaded.get("output_dim", output_dim),
        )
        model.load_state_dict(loaded["model_state_dict"])
        return model
    if isinstance(loaded, OrderedDict):
        model = AimStackedMLP(input_dim=input_dim, output_dim=output_dim)
        model.load_state_dict(loaded)
        return model
    return loaded

AIM_MODEL_CONFIG = {
    "MODEL_PATH": os.path.join(AIM_DIR, "aim_stacked_mlp_12.pth"),
    "INPUT_DIM" : 20 * SEQ_LEN,
    "OUTPUT_DIM": 2
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
    OBS_SIZE = 20  # Observation vector size
    BODY_LENGTH = 4 + 8 + 4 * OBS_SIZE  # int32 seq + int64 nanotime + float32 observation vector
    PACKET_LENGTH = 4 + BODY_LENGTH  # int32 body_length number + body

    # Header: body length
    header = recv_exact(conn, 4)
    body_len = int.from_bytes(header, byteorder="little", signed=False)

    if body_len != BODY_LENGTH:
        raise ValueError("Body length mismatch")

    # Payload: int32 seq + int64 nanotime + float32 observation vector
    payload = recv_exact(conn, body_len)
    seq = int.from_bytes(payload[:4], byteorder="little", signed=False)
    nanotime = int.from_bytes(payload[4:12], byteorder="little", signed=False)
    obs_bytes = payload[12:]
    obs = np.frombuffer(obs_bytes, dtype="<f4")
    return [seq, nanotime, obs]


def get_aim_data(data: np.ndarray):
    return data[:20]

def build_action_packet(seq: int, actions: list):
    actions = np.asarray(actions, dtype="<f4").reshape(-1)
    ACTION_SIZE = len(actions)  # Action vector size
    BODY_LENGTH = 4 + 8 + 4 * ACTION_SIZE  # int32 seq + int64 nanotime + float32 action vector
    PACKET_LENGTH = 4 + BODY_LENGTH  # int32 body_length number + body

    packet = bytearray()
    # Header: body length
    packet += BODY_LENGTH.to_bytes(4, byteorder="little", signed=False)
    # Payload: int32 seq + int64 nanotime + float32 action vector
    packet += seq.to_bytes(4, byteorder="little", signed=False)
    nanotime = time.time_ns()
    packet += nanotime.to_bytes(8, byteorder="little", signed=False)
    packet += actions.tobytes()

    return packet


def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"Server running at {host}/{port}.")

    while True:
        conn, addr = server.accept()
        print("Client connected", addr)
        aim_history = deque(maxlen=SEQ_LEN)
        aim_model = load_model(
            AIM_MODEL_CONFIG["MODEL_PATH"],
            AIM_MODEL_CONFIG["INPUT_DIM"],
            AIM_MODEL_CONFIG["OUTPUT_DIM"],
        )
        aim_model.to(device)
        aim_model.eval()

        try:
            while True:
                seq, nanotime, obs = read_obs_packet(conn)

                # Aiming
                aim_x = get_aim_data(obs)
                aim_history.append(aim_x)
                while len(aim_history) < SEQ_LEN:
                    aim_history.appendleft(aim_x)

                aim_tensor = torch.tensor(np.array(aim_history), dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    aim_y = aim_model(aim_tensor).reshape(-1).cpu().numpy()[:2]

                packet = build_action_packet(seq, aim_y)
                conn.sendall(packet)
        except ConnectionError as e:
            print("Client disconnected:", e)
        finally:
            conn.close()

if __name__ == "__main__":
    run_server()
