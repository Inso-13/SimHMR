# SimHMR: Simple Human Mesh Recovery

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

SimHMR is a simple and effective framework for 3D human mesh recovery from single images. It combines the power of transformer architectures with SMPL body model to achieve state-of-the-art performance on various benchmarks.

## Installation

### Prerequisites

- Python 3.8
- CUDA 10.2+ (for GPU training)
- Conda (recommended for environment management)

### Option 1: Using Conda Environment (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/your-username/simhmr.git
cd simhmr
```

2. Create and activate conda environment:
```bash
conda env create -f env.yml
conda activate human
```

3. Install SimHMR:
```bash
pip install -e .
```

### Option 2: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/simhmr.git
cd simhmr
```

2. Install PyTorch (CUDA 10.2):
```bash
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

4. Install SimHMR:
```bash
pip install -e .
```

### Key Dependencies

- **PyTorch**: 1.8.0
- **MMCV**: 1.5.3
- **MMDetection**: 2.27.0
- **MMPose**: 0.28.1
- **SMPL-X**: 0.1.28
- **PyTorch3D**: 0.7.2
- **OpenCV**: 4.7.0.68

## Quick Start

### Training

1. Prepare your dataset and update the configuration file
2. Start training:
```bash
python tools/train.py configs/simhmr/pw3d.py
```

### Evaluation

```bash
python tools/test.py configs/simhmr/pw3d.py work_dirs/checkpoint.pth --eval mpjpe pa-mpjpe
```

## Datasets

Please organise all datasets under the `data/` directory following the MMHuman3D format. See [MMHuman3D documentation](https://mmhuman3d.readthedocs.io/en/latest/data_preparation.html) for details.

## Citation

If you find this work useful, please cite:

```bibtex
@article{simhmr2024,
  title={SimHMR: Simple Human Mesh Recovery},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work builds upon [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)
- Thanks to the SMPL model authors
- Thanks to all the dataset providers

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.