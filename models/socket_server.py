# Python TCP server to transfer data between forge client and models

import socket
import json
import torch
import numpy as np

models = {
    "aim": torch.load("models/aim/aim_model_unscaled2.pth", map_location="cpu"),
}

for m in models.values():
    m.eval()

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 33333))
server.listen(1)

conn, addr = server.accept()
print("Client connected", addr)

# Data processing
def get_aim_data(data: dict):
    entries_arr = np.array(data.keys())
    field_arr = np.array(data.values())
    field_arr["target" in entries_arr] = 0
    target_entries = ["dx", "dy", "dz", "targetVelX", "targetVelY", "targetVelZ", "yawDiff", "pitchDiff"]
    return field_arr[entries_arr in target_entries]


while True:
    raw = conn.recv(4096).decode()
    if not raw:
        break

    request = json.loads(raw)

    # Aiming
    aim_model = models["aim"]
    aim_x = get_aim_data(request)
    aim_tensor = torch.tensor(aim_x).float().unsqueeze(0)

    with torch.no_grad():
        aim_y = aim_model(aim_tensor).sqeeze().tolist()

    conn.send(json.dumps(aim_y).encode())
