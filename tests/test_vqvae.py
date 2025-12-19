import torch
import sys
import os
import pytest

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqvae import VQVAE

def test_vqvae():
    print("Testing VQ-VAE...")
    
    # 1. Standard Configuration (128x128, 3 layers -> 16x16)
    print("\n1. Testing Standard Config (128x128, 3 layers)...")
    model = VQVAE(in_channels=3, base_channels=64, layers=2, num_downsamples=3, input_resolution=128, block_type='resnet')
    x = torch.randn(1, 3, 128, 128)
    recon, indices, loss = model(x)
    print(f"   Input: {x.shape}, Recon: {recon.shape}, Indices: {indices.shape}")
    assert recon.shape == x.shape
    assert indices.shape[1:] == (16, 16)

    # 2. Smaller Config (64x64, 2 layers -> 16x16)
    print("\n2. Testing Smaller Config (64x64, 2 layers)...")
    model = VQVAE(in_channels=3, base_channels=64, layers=2, num_downsamples=2, input_resolution=64, block_type='convnext')
    x = torch.randn(1, 3, 64, 64)
    recon, indices, loss = model(x)
    print(f"   Input: {x.shape}, Recon: {recon.shape}, Indices: {indices.shape}")
    assert recon.shape == x.shape
    assert indices.shape[1:] == (16, 16)

    # 3. Invalid Config (64x64, 3 layers -> 8x8 != 16x16)
    print("\n3. Testing Invalid Config (64x64, 3 layers)...")
    try:
        model = VQVAE(in_channels=3, base_channels=64, layers=2, num_downsamples=3, input_resolution=64)
        print("   FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"   PASSED: Caught expected error: {e}")

    # 4. Test Channel Multiplier
    print("\n4. Testing Channel Multiplier (1.5)...")
    model = VQVAE(input_resolution=64, num_downsamples=2, base_channels=16, channel_multiplier=1.5)
    x = torch.randn(1, 3, 64, 64)
    recon, indices, _ = model(x)
    print(f"   Input: {x.shape}, Recon: {recon.shape}, Indices: {indices.shape}")
    assert recon.shape == x.shape, f"Recon shape mismatch: {recon.shape} != {x.shape}"

    print("\nVQ-VAE Tests Passed!")

if __name__ == "__main__":
    test_vqvae()
