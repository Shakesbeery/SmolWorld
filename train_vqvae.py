import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models.vqvae import VQVAE

from tqdm import tqdm

class VPTDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading data from {data_path}...")
        data = torch.load(data_path)
        self.states = data['states'] # (T, H, W, C) in uint8
        print(f"Loaded {len(self.states)} frames.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Lazy processing: Convert to float and permute on the fly
        # Input: (H, W, C) uint8
        # Output: (C, H, W) float32 normalized
        state = self.states[idx]
        return state.permute(2, 0, 1).float() / 255.0

def save_sample(model, x, epoch, output_dir):
    model.eval()
    with torch.no_grad():
        recon, _, _ = model(x)
    
    # Take first 8 samples
    n = min(x.size(0), 8)
    x = x[:n].cpu()
    recon = recon[:n].cpu()
    
    # Create grid
    fig, axes = plt.subplots(2, n, figsize=(n*2, 4))
    for i in range(n):
        # Input
        axes[0, i].imshow(x[i].permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Input")
        
        # Recon
        axes[1, i].imshow(np.clip(recon[i].permute(1, 2, 0).numpy(), 0, 1))
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Recon")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_epoch_{epoch}.png"))
    plt.close()

def plot_loss(losses, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Trend')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE")
    parser.add_argument("--data_path", type=str, required=True, help="Path to processed .pt file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--resolution", type=int, default=64, help="Input resolution")
    parser.add_argument("--downsamples", type=int, default=2, help="Number of downsample layers")
    parser.add_argument("--base_channels", type=int, default=32, help="Base channel dimension (hidden_dim)")
    parser.add_argument("--codebook_size", type=int, default=1024, help="Size of the codebook")
    parser.add_argument("--codebook_dim", type=int, default=256, help="Dimension of codebook embeddings")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    dataset = VPTDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Model
    model = VQVAE(
        in_channels=3, 
        base_channels=args.base_channels, 
        layers=2, 
        num_downsamples=args.downsamples,
        input_resolution=args.resolution,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim
    ).to(device)
    print("Model initialized")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=args.patience//2)
    
    # Training Loop
    best_loss = float('inf')
    patience_counter = 0
    epoch_losses = []
    
    print("Beginning training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            x = batch.to(device)
            
            optimizer.zero_grad()
            recon, _, commit_loss = model(x)
            
            recon_loss = nn.MSELoss()(recon, x)
            loss = recon_loss + commit_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f}")
        
        # Scheduler
        scheduler.step(avg_loss)
        
        # Visualization
        save_sample(model, batch[:8].to(device), epoch+1, args.output_dir)
        plot_loss(epoch_losses, args.output_dir)
        
        # Early Stopping & Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print("  Saved best model.")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break
            
    print("Training complete.")

if __name__ == "__main__":
    main()
