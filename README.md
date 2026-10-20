# GRIP

Official PyTorch implementation of GRIP: 
From Generation to Grounding: Real-Image Proxies for Zero-Shot Composed Image Retrieval.

## Data preparation

### CIRCO CIRR and FashionIQ

Download the CIRCO dataset following the instructions in the https://github.com/miccunifi/CIRCO.

Download the CIRR dataset following the instructions in the https://github.com/Cuberick-Orion/CIRR.

Download the FashionIQ dataset following the instructions in the https://github.com/XiaoxiaoGuo/fashion-iq.
## Conda environment setup

The CIRCO pipeline uses Python 3.12.3, PyTorch 2.2.2, and torchvision 0.17.2 with CUDA 12.1. GPU execution requires a Linux system with an NVIDIA GPU and a compatible driver.

Run the following commands from the repository root:

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate grip

# Check installed dependencies
python -m pip check

# Check PyTorch and CUDA availability
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

