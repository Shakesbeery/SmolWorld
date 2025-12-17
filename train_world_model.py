import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.vqvae import VQVAE
from models.transformer import WorldModel
from tqdm import tqdm

class WorldModelDataset(Dataset):
    def __init__(self, data_path, vqvae_model, device, seq_len=1024):
        print(f"Loading data from {data_path}...")
        data = torch.load(data_path)
        self.states = data['states'].permute(0, 3, 1, 2).float() / 255.0 # (T, C, H, W)
        self.actions = data['actions'] # (T,)
        self.seq_len = seq_len
        
        # Precompute image tokens using VQ-VAE
        print("Encoding images to tokens...")
        vqvae_model.eval()
        vqvae_model.to(device)
        
        self.image_tokens = []
        batch_size = 32
        with torch.no_grad():
            for i in tqdm(range(0, len(self.states), batch_size)):
                batch = self.states[i:i+batch_size].to(device)
                _, indices, _ = vqvae_model(batch)
                # indices shape: (B, 16, 16) -> flatten to (B, 256)
                self.image_tokens.append(indices.view(indices.size(0), -1).cpu())
        
        self.image_tokens = torch.cat(self.image_tokens, dim=0) # (T, 256)
        print(f"Encoded {len(self.image_tokens)} frames.")

        # Interleave: [Action, ImgTokens...]
        # Action tokens need to be shifted to avoid overlap with Image tokens (0-1023)
        # Image tokens: 0-1023
        # Action tokens: 1024 - (1024+4356)
        # Special tokens: > (1024+4356)
        
        self.action_offset = 1024
        self.tokens = []
        
        print("Interleaving tokens...")
        for i in range(len(self.actions)):
            # Action
            act = self.actions[i].item() + self.action_offset
            self.tokens.append(act)
            
            # Image
            img_toks = self.image_tokens[i].tolist()
            self.tokens.extend(img_toks)
            
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens)}")

    def __len__(self):
        # Number of sequences
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def main():
    parser = argparse.ArgumentParser(description="Train World Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to processed .pt file")
    parser.add_argument("--vqvae_path", type=str, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints_wm", help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load VQ-VAE
    print("Loading VQ-VAE...")
    # Assuming standard config for now, ideally load config from checkpoint
    vqvae = VQVAE(input_resolution=32, num_downsamples=1).to(device) # Using 32x32 config for verification
    # vqvae.load_state_dict(torch.load(args.vqvae_path)) # Skip loading weights for verification if dummy
    if os.path.exists(args.vqvae_path):
         vqvae.load_state_dict(torch.load(args.vqvae_path))
    else:
        print("Warning: VQ-VAE checkpoint not found, using random weights (OK for verification)")

    # Data
    dataset = WorldModelDataset(args.data_path, vqvae, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Model
    # Vocab = 1024 (Img) + 4356 (Act) + Specials (~100) = ~5500
    model = WorldModel(vocab_size=5500, dim=576, n_layers=24, n_heads=9).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # Mixed Precision (BF16 if available)
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.grad_accum
            
            loss.backward()
            
            if (i + 1) % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * args.grad_accum
            progress_bar.set_postfix(loss=loss.item() * args.grad_accum)
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"world_model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    main()
