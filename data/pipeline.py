import os
import argparse
import requests
import tarfile
import json
import numpy as np
import cv2
import torch
import random
from tqdm import tqdm
from huggingface_hub import list_repo_files

# Configuration
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "open-world-agents/vpt-owamcap"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
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
    temp_path = os.path.join(DATA_DIR, f"temp_pipeline_{random.randint(0, 100000)}.mp4")
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

def process_shard(shard_path, resolution):
    print(f"Processing shard: {shard_path}")
    all_states = []
    all_actions = []
    
    with tarfile.open(shard_path, "r") as tar:
        members = tar.getmembers()
        files = {}
        for m in members:
            if m.isfile():
                base = os.path.splitext(m.name)[0]
                ext = os.path.splitext(m.name)[1]
                if base not in files: files[base] = {}
                files[base][ext] = m
        
        for base, items in tqdm(files.items(), desc="Processing episodes"):
            if ".mp4" in items and ".jsonl" in items:
                try:
                    # Process Video
                    f_vid = tar.extractfile(items[".mp4"])
                    vid_bytes = f_vid.read()
                    frames = process_video(vid_bytes, resolution)
                    
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
        return None, None

    final_states = np.concatenate(all_states, axis=0)
    final_actions = np.concatenate(all_actions, axis=0)
    return final_states, final_actions

def main():
    parser = argparse.ArgumentParser(description="VPT Data Pipeline")
    parser.add_argument("--resolution", type=int, default=64, help="Target video resolution (square)")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to download and process")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of raw data")
    args = parser.parse_args()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # 1. List and Sample Shards
    print("Fetching file list from Hugging Face...")
    files = list_repo_files(REPO_ID, repo_type="dataset")
    tar_files = [f for f in files if f.endswith(".tar")]
    print(f"Found {len(tar_files)} available shards.")
    
    selected_shards = random.sample(tar_files, min(args.num_shards, len(tar_files)))
    print(f"Selected shards: {selected_shards}")

    for shard_name in selected_shards:
        raw_path = os.path.join(DATA_DIR, shard_name)
        processed_filename = f"{os.path.splitext(shard_name)[0]}_res{args.resolution}.pt"
        processed_path = os.path.join(PROCESSED_DIR, processed_filename)
        
        # Check if processed exists
        if os.path.exists(processed_path) and not args.force_download:
            print(f"Processed file {processed_path} already exists. Skipping.")
            continue

        # Download
        if args.force_download or not os.path.exists(raw_path):
            url = f"{BASE_URL}/{shard_name}"
            download_file(url, raw_path)
        else:
            print(f"Found existing raw data at {raw_path}. Skipping download.")

        # Process
        states, actions = process_shard(raw_path, args.resolution)
        
        if states is not None:
            print(f"Saving {len(states)} frames to {processed_path}...")
            torch.save({
                'states': torch.from_numpy(states),
                'actions': torch.from_numpy(actions)
            }, processed_path)
            print("Done!")
        else:
            print(f"No valid data found in {shard_name}.")
        
        # Optional: Clean up raw file to save space? 
        # For now, keeping it as per original behavior, but user might want to delete it.
        # os.remove(raw_path)

if __name__ == "__main__":
    main()
