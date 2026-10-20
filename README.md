# GRIP

Official PyTorch implementation of GRIP: From Generation to Grounding: Real-Image Proxies for Zero-Shot Composed Image Retrieval.

## Data preparation

### CIRCO

Download the CIRCO dataset following the instructions in the https://github.com/miccunifi/CIRCO.

### CIRR

Download the CIRR dataset following the instructions in the https://github.com/Cuberick-Orion/CIRR.

### FashionIQ

Download the FashionIQ dataset following the instructions in the https://github.com/XiaoxiaoGuo/fashion-iq.
## Conda environment setup

The CIRCO pipeline uses Python 3.12.3, PyTorch 2.2.2, and
torchvision 0.17.2 with CUDA 12.1. GPU execution requires a Linux
system with an NVIDIA GPU and a compatible driver.

### Create the environment

Run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate grip
```

### Verify the installation

Check the installed dependencies:

```bash
python -m pip check
```

Check PyTorch and CUDA availability:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

For a correctly configured GPU environment, the expected output is:

```text
PyTorch: 2.2.2+cu121
CUDA: 12.1
GPU available: True
```
