import os
import argparse
import requests
import tarfile
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Configuration
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
# Correct URL we found earlier
DATASET_URL = "https://huggingface.co/datasets/open-world-agents/vpt-owamcap/resolve/main/shard-000000.tar"
RAW_SHARD_PATH = os.path.join(DATA_DIR, "raw_shard.tar")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MOUSE_BINS = 11
MOUSE_CLIP = 20

def download_file(url, filename):
    print(f"Downloading from {url} to {filename}...")
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    print("Download complete.")

def quantize_mouse(dx, dy):
    dx = np.clip(dx, -MOUSE_CLIP, MOUSE_CLIP)
    dy = np.clip(dy, -MOUSE_CLIP, MOUSE_CLIP)
    x_bin = int((dx + MOUSE_CLIP) / (2 * MOUSE_CLIP) * (MOUSE_BINS - 1))
    y_bin = int((dy + MOUSE_CLIP) / (2 * MOUSE_CLIP) * (MOUSE_BINS - 1))
    return x_bin, y_bin

def encode_action(action_dict):
    camera = action_dict.get('camera', [0, 0])
    dx, dy = camera[0], camera[1]
    attack = action_dict.get('attack', 0)
    jump = action_dict.get('jump', 0)
    
    forward = action_dict.get('forward', 0)
    back = action_dict.get('back', 0)
    left = action_dict.get('left', 0)
    right = action_dict.get('right', 0)
    
    move_idx = 0
    if forward and not back:
        if left: move_idx = 5
        elif right: move_idx = 6
        else: move_idx = 1
    elif back and not forward:
        if left: move_idx = 7
        elif right: move_idx = 8
        else: move_idx = 3
    elif left: move_idx = 2
    elif right: move_idx = 4
        
    mx, my = quantize_mouse(dx, dy)
    
    idx = 0
    stride = 1
    idx += mx * stride; stride *= MOUSE_BINS
    idx += my * stride; stride *= MOUSE_BINS
    idx += attack * stride; stride *= 2
    idx += jump * stride; stride *= 2
    idx += move_idx * stride
    
    return idx

def process_video(video_bytes, resolution):
    temp_path = os.path.join(DATA_DIR, "temp_pipeline.mp4")
    with open(temp_path, "wb") as f:
        f.write(video_bytes)
        
    cap = cv2.VideoCapture(temp_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (resolution, resolution))
        frames.append(frame)
    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return np.array(frames, dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description="VPT Data Pipeline")
    parser.add_argument("--resolution", type=int, default=64, help="Target video resolution (square)")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of raw data")
    args = parser.parse_args()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Download
    if args.force_download or not os.path.exists(RAW_SHARD_PATH):
        download_file(DATASET_URL, RAW_SHARD_PATH)
    else:
        print(f"Found existing raw data at {RAW_SHARD_PATH}. Skipping download.")

    # 2. Preprocess
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    print(f"Processing data with resolution {args.resolution}x{args.resolution}...")
    
    all_states = []
    all_actions = []
    
    with tarfile.open(RAW_SHARD_PATH, "r") as tar:
        members = tar.getmembers()
        files = {}
        for m in members:
            if m.isfile():
                base = os.path.splitext(m.name)[0]
                ext = os.path.splitext(m.name)[1]
                if base not in files: files[base] = {}
                files[base][ext] = m
        
        for base, items in tqdm(files.items()):
            if ".mp4" in items and ".jsonl" in items:
                try:
                    # Process Video
                    f_vid = tar.extractfile(items[".mp4"])
                    vid_bytes = f_vid.read()
                    frames = process_video(vid_bytes, args.resolution)
                    
                    # Process Actions
                    f_act = tar.extractfile(items[".jsonl"])
                    actions = []
                    for line in f_act:
                        act = json.loads(line)
                        actions.append(encode_action(act))
                    actions = np.array(actions, dtype=np.int64)
                    
                    min_len = min(len(frames), len(actions))
                    frames = frames[:min_len]
                    actions = actions[:min_len]
                    
                    all_states.append(frames)
                    all_actions.append(actions)
                except Exception as e:
                    print(f"Error processing {base}: {e}")
                    continue

    if not all_states:
        print("No valid data processed.")
        return

    final_states = np.concatenate(all_states, axis=0)
    final_actions = np.concatenate(all_actions, axis=0)
    
    output_filename = f"shard_res{args.resolution}.pt"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    
    print(f"Saving {len(final_states)} frames to {output_path}...")
    torch.save({
        'states': torch.from_numpy(final_states),
        'actions': torch.from_numpy(final_actions)
    }, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
