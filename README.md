# SmolWorld: A Minimalist World Model

SmolWorld is a research repository for training Generative World Models on the OpenAI VPT (Video PreTraining) dataset. It implements a two-stage pipeline:
1.  **Visual Tokenizer (VQ-VAE)**: Compresses video frames into discrete tokens.
2.  **World Model (Transformer)**: Predicts the next visual tokens autoregressively based on history and actions.

## 🚀 Technology Stack

*   **VQ-VAE**:
    *   **Architecture**: CNN Encoder/Decoder with ResNet or ConvNeXt blocks.
    *   **Quantization**: Vector Quantization with learnable codebook (custom implementation).
    *   **Config**: 16x16 latent grid, 1024 codebook size, 256 dim.
*   **World Model**:
    *   **Architecture**: Decoder-only Transformer (Llama-style).
    *   **Features**: RMSNorm (Pre-norm), RoPE (Rotary Positional Embeddings), SwiGLU activation.
    *   **Config**: ~150M params (24 layers, 576 dim, 9 heads), 1024 context window.
*   **Training**:
    *   **Precision**: BF16/FP16 Mixed Precision.
    *   **Optimization**: AdamW with ReduceLROnPlateau.
    *   **Data**: Interleaved Action/Image tokens.

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/SmolWorld.git
    cd SmolWorld
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r pyproject.toml
    # Or manually:
    pip install torch numpy opencv-python-headless einops vector-quantize-pytorch flash-attn requests tqdm pandas matplotlib
    ```

## 🛠️ Usage

### 1. Data Pipeline
Download and preprocess the OpenAI VPT dataset.

```bash
# Download and process to 64x64 resolution
python data/pipeline.py --resolution 64

# Force re-download if needed
python data/pipeline.py --resolution 64 --force-download
```
*Output*: `data/processed/shard_res64.pt`

### 2. Train Visual Tokenizer (VQ-VAE)
Train the VQ-VAE to compress images into tokens.

```bash
# Train on 64x64 data
python train_vqvae.py --data_path data/processed/shard_res64.pt --resolution 64 --downsamples 2 --batch_size 32 --epochs 100

# Resume/Fine-tune
python train_vqvae.py --data_path ... --epochs 50
```
*Output*: `checkpoints/best_model.pt` (and visualization images)

### 3. Train World Model
Train the Transformer to predict the future.

```bash
# Train using the trained VQ-VAE
python train_world_model.py --data_path data/processed/shard_res64.pt --vqvae_path checkpoints/best_model.pt --epochs 50 --batch_size 4 --grad_accum 4
```
*Output*: `checkpoints_wm/world_model_epoch_X.pt`

### 4. Play the Dream (Inference)
Interactively generate future frames based on your input.

```bash
# Interactive Mode (Requires GUI)
# Controls: 'W' to move forward, 'Q' to quit.
python inference/play.py --vqvae_path checkpoints/best_model.pt --wm_path checkpoints_wm/world_model_epoch_50.pt --resolution 64 --downsamples 2

# Headless Mode (Save frames to disk)
python inference/play.py --vqvae_path ... --wm_path ... --headless --steps 100
```

## 📂 Repository Structure

```
SmolWorld/
├── data/
│   ├── pipeline.py       # Unified download & preprocess script
│   └── processed/        # Processed .pt shards
├── models/
│   ├── vqvae.py          # VQ-VAE Architecture
│   ├── quantizer.py      # Custom Vector Quantizer
│   └── transformer.py    # World Model (Llama-style)
├── inference/
│   └── play.py           # Interactive inference script
├── tests/                # Unit tests
├── train_vqvae.py        # VQ-VAE Training Script
├── train_world_model.py  # World Model Training Script
├── pyproject.toml        # Dependencies
└── README.md             # Documentation
```

## 📝 Action Encoding
Actions are mapped to a single integer (0-4355):
*   **Mouse**: Quantized into 11x11 bins.
*   **Keys**: WASD mapped to 9 states.
*   **Binary**: Attack (L-Click), Jump (Space).
*   **Formula**: `idx = mx + my*11 + attack*121 + jump*242 + move*484`
