import torch
import os

os.makedirs("data/processed", exist_ok=True)

# Create dummy data: 100 frames, 32x32, 3 channels
states = torch.randint(0, 255, (100, 32, 32, 3), dtype=torch.uint8)
actions = torch.zeros(100, dtype=torch.int64)

torch.save({
    'states': states,
    'actions': actions
}, "data/processed/dummy_ema.pt")
print("Created data/processed/dummy_ema.pt")
