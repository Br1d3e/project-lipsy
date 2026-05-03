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
sys.path.insert(0, AIM_DIR)

from train_aim_model import AimLSTM

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEQ_LEN = 20

def load_model(path: str):
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, OrderedDict):
        model = AimLSTM(input_dim=9, hidden_dim=128, num_layers=2)
        model.load_state_dict(loaded)
        return model
    return loaded

models = {
    "aim": load_model(os.path.join(AIM_DIR, "aim_model_unscaled2.pth")),
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
    OBS_SIZE = 9  # Observation vector size
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
    return np.array(data[:9])

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
    for m in models.values():
        m.to(device)
        m.eval()

    aim_history = deque(maxlen=SEQ_LEN)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"Server running at {host}/{port}.")

    while True:
        conn, addr = server.accept()
        print("Client connected", addr)
        try:
            while True:
                seq, nanotime, obs = read_obs_packet(conn)

                # Aiming
                aim_model = models["aim"]
                aim_x = get_aim_data(obs)
                aim_history.append(aim_x)
                while len(aim_history) < SEQ_LEN:
                    aim_history.appendleft(aim_x)

                aim_tensor = torch.tensor(np.array(aim_history), dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    aim_y = aim_model(aim_tensor).squeeze().cpu().tolist()

                packet = build_action_packet(seq, aim_y)
                conn.sendall(packet)
        except ConnectionError as e:
            print("Client disconnected:", e)
        finally:
            conn.close()

if __name__ == "__main__":
    run_server()
