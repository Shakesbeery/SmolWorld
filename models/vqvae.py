import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import VectorQuantizer

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)

class ConvNextBlock(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.GroupNorm(1, dim * mult),
            nn.Conv2d(dim * mult, dim, 1)
        )

    def forward(self, x):
        residual = x
        x = self.ds_conv(x)
        x = self.net(x)
        return residual + x

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, layers, num_downsamples, block_type='resnet'):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Initial projection
        self.layers.append(nn.Conv2d(in_channels, hidden_dim, 3, padding=1))
        
        current_dim = hidden_dim
        
        # Downsampling layers
        for _ in range(num_downsamples):
            # Add blocks
            for _ in range(layers):
                if block_type == 'resnet':
                    self.layers.append(ResNetBlock(current_dim))
                else:
                    self.layers.append(ConvNextBlock(current_dim))
            
            # Downsample and double channels
            self.layers.append(nn.Conv2d(current_dim, current_dim * 2, 4, stride=2, padding=1))
            self.layers.append(nn.ReLU() if block_type == 'resnet' else nn.GELU())
            current_dim *= 2

        # Final blocks
        for _ in range(layers):
            if block_type == 'resnet':
                self.layers.append(ResNetBlock(current_dim))
            else:
                self.layers.append(ConvNextBlock(current_dim))
                
        self.final_conv = nn.Conv2d(current_dim, 256, 1) # Project to latent dim

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)

class Decoder(nn.Module):
    def __init__(self, out_channels, hidden_dim, layers, num_upsamples, block_type='resnet'):
        super().__init__()
        
        # Calculate starting dimension (reverse of encoder)
        current_dim = hidden_dim * (2 ** num_upsamples)
        
        self.initial_conv = nn.Conv2d(256, current_dim, 1)
        self.layers = nn.ModuleList()
        
        # Initial blocks
        for _ in range(layers):
            if block_type == 'resnet':
                self.layers.append(ResNetBlock(current_dim))
            else:
                self.layers.append(ConvNextBlock(current_dim))
                
        # Upsampling layers
        for _ in range(num_upsamples):
            # Upsample and halve channels
            self.layers.append(nn.ConvTranspose2d(current_dim, current_dim // 2, 4, stride=2, padding=1))
            self.layers.append(nn.ReLU() if block_type == 'resnet' else nn.GELU())
            current_dim //= 2
            
            # Add blocks
            for _ in range(layers):
                if block_type == 'resnet':
                    self.layers.append(ResNetBlock(current_dim))
                else:
                    self.layers.append(ConvNextBlock(current_dim))
                    
        self.final_conv = nn.Conv2d(current_dim, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)

class VQVAE(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 hidden_dim=128, 
                 layers=2, 
                 num_downsamples=3,
                 input_resolution=128,
                 block_type='convnext',
                 codebook_size=1024,
                 codebook_dim=256,
                 decay=0.99):
        super().__init__()
        
        # Validation
        downsample_factor = 2 ** num_downsamples
        expected_latent_size = 16
        if input_resolution // downsample_factor != expected_latent_size:
            raise ValueError(
                f"Invalid configuration: input_resolution ({input_resolution}) / "
                f"downsample_factor ({downsample_factor}) != {expected_latent_size}. "
                f"Result is {input_resolution / downsample_factor}."
            )
        
        self.encoder = Encoder(in_channels, hidden_dim, layers, num_downsamples, block_type)
        self.vq = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=codebook_dim,
            commitment_cost=0.25,
            decay=decay
        )
        self.decoder = Decoder(in_channels, hidden_dim, layers, num_downsamples, block_type)

    def forward(self, x):
        z = self.encoder(x)
        quantized, indices, commit_loss = self.vq(z)
        recon = self.decoder(quantized)
        
        return recon, indices, commit_loss
