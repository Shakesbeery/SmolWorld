import torch
import sys
import os

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import WorldModel

def test_transformer():
    print("Testing World Model Transformer...")
    
    vocab_size = 5500
    dim = 576
    n_layers = 2
    n_heads = 9
    seq_len = 128
    
    model = WorldModel(vocab_size=vocab_size, dim=dim, n_layers=n_layers, n_heads=n_heads, max_seq_len=1024)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")
    
    # Dummy input
    x = torch.randint(0, vocab_size, (1, seq_len))
    
    # Forward
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    assert logits.shape == (1, seq_len, vocab_size)
    print("Transformer Tests Passed!")

if __name__ == "__main__":
    test_transformer()
