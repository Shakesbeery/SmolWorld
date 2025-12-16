# World Model Experiments

Personal experiments with world models using the OpenAI VPT dataset.

## Setup

1. Install dependencies:
   ```bash
   pip install -r pyproject.toml
   # Note: flash-attn might need manual installation on Windows.
   ```

2. Run Data Pipeline:
   ```bash
   # Download and process with default resolution (64x64)
   python data/pipeline.py
   
   # Process with custom resolution (e.g., 128x128)
   python data/pipeline.py --resolution 128
   
   # Force re-download if needed
   python data/pipeline.py --force-download
   ```
   This script will:
   - Download the data shard if missing.
   - Resize videos to the specified resolution.
   - Quantize actions.
   - Save processed data to `data/processed/shard_res{resolution}.pt`.

## Data Structure

The processed data is saved as a PyTorch file containing a dictionary:
- `states`: Tensor of shape (T, 64, 64, 3) (uint8)
- `actions`: Tensor of shape (T,) (int64)

## Action Space

The action space is a flattened integer representation of:
- Mouse X (11 bins)
- Mouse Y (11 bins)
- Attack (Binary)
- Jump (Binary)
- Movement (Stop, W, A, S, D, WA, WD, SA, SD)

Total action space size: 4356.
