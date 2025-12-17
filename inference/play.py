import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqvae import VQVAE
from models.transformer import WorldModel

def get_action_token(key_code):
    # Mapping:
    # Mouse: 5, 5 (Center) -> 5 + 5*11 = 60
    # Attack: 0
    # Jump: 0
    # Move: Stop(0), W(1), A(2), S(3), D(4)
    # Token = mx + my*11 + attack*121 + jump*242 + move*484
    
    # Base (Center mouse, no attack/jump)
    base = 5 + 5 * 11 # 60
    
    move_idx = 0
    if key_code == ord('w'): move_idx = 1
    elif key_code == ord('a'): move_idx = 2
    elif key_code == ord('s'): move_idx = 3
    elif key_code == ord('d'): move_idx = 4
    
    token = base + move_idx * 484
    
    # Offset by 1024 (Image vocab size)
    return token + 1024

def main():
    parser = argparse.ArgumentParser(description="Play the Dream")
    parser.add_argument("--vqvae_path", type=str, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument("--wm_path", type=str, required=True, help="Path to World Model checkpoint")
    parser.add_argument("--resolution", type=int, default=64, help="Resolution")
    parser.add_argument("--downsamples", type=int, default=2, help="Downsamples")
    
    parser.add_argument("--n_layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--dim", type=int, default=576, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=9, help="Number of heads")
    
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to run in headless mode")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Output directory for headless mode")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Models
    print("Loading models...")
    vqvae = VQVAE(input_resolution=args.resolution, num_downsamples=args.downsamples).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=device))
    vqvae.eval()
    
    wm = WorldModel(vocab_size=5500, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    wm.load_state_dict(torch.load(args.wm_path, map_location=device))
    wm.eval()
    
    # Initialize Context
    # Start with blank image tokens (e.g., all zeros)
    # 16x16 = 256 tokens
    context = torch.zeros(1, 256, dtype=torch.long).to(device)
    
    if args.headless:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Running in headless mode for {args.steps} steps...")
    else:
        print("Starting simulation... Press 'q' to quit, 'w' to move forward.")
        cv2.namedWindow("Dream", cv2.WINDOW_NORMAL)
    
    step = 0
    while True:
        if args.headless:
            if step >= args.steps:
                break
            key = ord('w') # Simulate moving forward
            step += 1
        else:
            # 1. Get User Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
        action_token = get_action_token(key)
        
        # 2. Append Action
        action_tensor = torch.tensor([[action_token]], device=device)
        context = torch.cat([context, action_tensor], dim=1)
        
        # 3. Generate Next Frame (256 tokens)
        # We need to generate 256 tokens autoregressively
        # This is slow, but necessary for "dreaming"
        
        generated_frame = []
        
        print(f"Generating frame {step}...", end="\r")
        for _ in range(256):
            # Crop context to max_seq_len if needed
            if context.size(1) > 1024:
                context = context[:, -1024:]
                
            with torch.no_grad():
                logits = wm(context)
                next_token_logits = logits[:, -1, :]
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                # Mask out action tokens (>= 1024) to ensure we generate image tokens
                probs[:, 1024:] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)
                
                context = torch.cat([context, next_token], dim=1)
                generated_frame.append(next_token)
        
        # 4. Decode and Display
        frame_tokens = torch.cat(generated_frame, dim=1) # (1, 256)
        indices = frame_tokens.view(1, 16, 16) # Reshape to grid
        
        with torch.no_grad():
            # Get codebook
            codebook = vqvae.vq._embedding.weight # (1024, 256)
            
            # Map indices to vectors
            vectors = F.embedding(indices, codebook)
            vectors = vectors.permute(0, 3, 1, 2) # (1, 256, 16, 16)
            
            recon = vqvae.decoder(vectors)
            
        # Convert to numpy image
        img = recon[0].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Upscale for display
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        if args.headless:
            cv2.imwrite(os.path.join(args.output_dir, f"frame_{step:04d}.png"), img)
        else:
            cv2.imshow("Dream", img)
    
    if not args.headless:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
