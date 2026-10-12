# Two-Directional X-ray Refraction Angle Imaging from a Single Absorption Measurement via Deep Learning
**two-directional-refraction** provides a deep learning framework based on pix2pix capable of translating one absorption image into both horizontal and vertical refraction angle images.
## Installation
1. **Clone the repo:**
   ```bash
   git clone https://github.com/ZhuYilan2238/two-directional-refraction.git
   cd two-directional-refraction
   ```
2. **Prepare the Conda environment:**
   ```bash
   conda create -n two-directional-refraction python=3.12
   conda activate two-directional-refraction
   pip install -r requirements.txt
   ```
## Usage
### Setup data
Organize the dataset as follows:
```text
datasets/
  ├── train/
  │   ├── H/
  │   └── V/
  ├── val/
  │   ├── H/
  │   └── V/
  └── test/
      ├── H/
      └── V/
```
### Train
```bash
python train.py --dataroot ./datasets --name two-directional-refraction --model pix2pix --netG nafnet
```
Validation runs by default. You can disable it with the option `--no_val`.
### Test
```bash
python test.py --dataroot ./datasets --name two-directional-refraction --model pix2pix --netG nafnet
```
## Acknowledgements
This project builds upon the foundational work of [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [NAFNet](https://github.com/megvii-research/NAFNet).
