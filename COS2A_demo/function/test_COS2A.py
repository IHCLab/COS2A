import torch
import torch.nn as nn
import os
import numpy as np
import scipy.io as sio
import time
import sys
from COS2A import *

scene_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_list_path = os.path.join(root_dir, 'data', 'data_list.txt')
with open(data_list_path, 'r') as f:
    scene_keys = [line.strip() for line in f.readlines()]
if scene_id < 1 or scene_id > len(scene_keys):
    print(f"Invalid scene_id {scene_id}, should be 1 to {len(scene_keys)}")
    exit()   
scene_key = scene_keys[scene_id - 1]
input_mat_path = os.path.join(root_dir, 'data', f'{scene_key}.mat')
output_mat_path = os.path.join(root_dir, 'DE_result', f'COS2A_result_{scene_key}.mat')
os.makedirs(os.path.dirname(output_mat_path), exist_ok=True)
model_path = os.path.join(root_dir, 'checkpoint', 'real_epoch_100_COS2A.pth')

gpu_id = '0'
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

mat = sio.loadmat(input_mat_path)
input_data = mat['Y'].astype(np.float32)
input_data = torch.from_numpy(input_data.transpose(2, 0, 1)).unsqueeze(0).cuda()

model = COS2A(in_channels=12, out_channels=172).cuda()
model.num_iterations = 2

checkpoint = torch.load(model_path)
try:
    if 'param' in checkpoint:
        model.load_state_dict(checkpoint['param'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()
with torch.no_grad():
    output = model(input_data)

output_np = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
sio.savemat(output_mat_path, {'output': output_np})

print(f"Data: {scene_key}")
print(f"Saved result to: {output_mat_path}")
