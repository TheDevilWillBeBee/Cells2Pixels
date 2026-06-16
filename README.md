# Neural Cellular Automata: From Cells to Pixels

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://cells2pixels.github.io/#growing)
[![arXiv](https://img.shields.io/badge/arXiv-2506.22899-b31b1b)](https://arxiv.org/abs/2506.22899v3)
![SIGGRAPH 2026](https://img.shields.io/badge/SIGGRAPH-2026-green)
[![Open Simple Texture In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheDevilWillBeBee/Cells2Pixels/blob/main/notebooks/simple_texture.ipynb)

![Teaser](data/repo/teaser.jpg)

Official implementation of **Neural Cellular Automata: From Cells to Pixels** (SIGGRAPH 2026).

## Installation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The pinned PyTorch version is chosen to match the Kaolin build in `requirements.txt` (`torch==2.8.0`, `kaolin==0.18.0`). Kaolin is only needed for experiments involving meshes, namely mesh rendering and rasterization, and the only Kaolin functionality used is `dibr_rasterization`. If you are not planning to run mesh experiments, you can use another compatible PyTorch version and skip Kaolin.

## Data

Download the datasets with:

```bash
python scripts/download_data.py
```

The script downloads and extracts the data expected by the configs:

- `data/morphology_png`: transparent PNG target images for 2D growing experiments.
- `data/textures_hr`: high-resolution 2D texture images for texture synthesis.
- `data/pbr_textures`: PBR texture maps. The downloader also combines height, roughness, and ambient occlusion maps into `hra.jpg`.
- `data/meshes`: OBJ meshes for mesh-based experiments.
- `data/textures_3d`: texture image targets for 3D texture experiments.
- `data/solid_textures`: volume texture files used by voxel experiments.
- `data/radiance_fields`: posed image datasets for radiance-field experiments.
- `data/projections`: cache directory used for generated mesh projections.
- `data/pretrained`: pretrained checkpoints, such as the optic-flow model and forthcoming NCA graft checkpoints used through `graft_initialization` to accelerate training.

## Training

All experiments use the same entry point:

```bash
python train.py --config <path-to-config>
```

For example:

```bash
python train.py --config configs/nca2d/growing.yaml
python train.py --config configs/nca2d/pbr_texture.yaml
python train.py --config configs/meshnca/texture.yaml
python train.py --config configs/nca3d/3d_texture.yaml
```

Paper training modes:

| Mode | Config | Notes |
| --- | --- | --- |
| Growing a 2D morphology | `configs/nca2d/growing.yaml` | Trains an NCA to grow a target RGBA image from a seed. |
| PBR texture synthesis | `configs/nca2d/pbr_texture.yaml` | Trains a 2D NCA to synthesize a PBR Texture. |
| Texture synthesis on meshes | `configs/meshnca/texture.yaml` | Trains MeshNCA to synthesize a texture directly on mesh surfaces. |
| Growing a 3D texture | `configs/nca3d/3d_texture.yaml` | Trains an NCA to synthesize a 3D volumetric texture. |

The following modes are exploratory experiments that are not included in the paper, but are included for curious readers:

| Mode | Config | Notes |
| --- | --- | --- |
| Dynamic textures | `configs/nca2d/dynamic_texture.yaml` | Trains a 2D texture NCA with an additional motion loss. |
| Growing a radiance field | `configs/nca3d/growing-radiance_field.yaml` | Trains a volumetric NCA to grow a radiance field from posed images. |
| Growing a voxel | `configs/nca3d/growing-voxel.yaml` | Trains a volumetric NCA to grow both a voxelized shape and its solid texture. |

Early growing-radiance-field result:

![Growing radiance field result](data/repo/rf-lego.gif)

To run the common test pass after training, add `--test`. This loads the saved checkpoint, rolls the NCA forward for `--test-steps`, and writes a rendered image and rollout video to `--test-output-dir`:

```bash
python train.py --config configs/nca2d/growing.yaml --test
```

Outputs are written under the configured experiment directory, and test images or videos are written to `outputs` by default.

You can also run simple 2D texture synthesis on `bubbly_0101.jpg` end-to-end in your browser via Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheDevilWillBeBee/Cells2Pixels/blob/main/notebooks/simple_texture.ipynb)

## Web Demo

To deploy trained models on the interactive web demos, see [Cells2Pixels/Cells2Pixels.github.io](https://github.com/Cells2Pixels/Cells2Pixels.github.io).

## TODO

- [ ] Add a Google Colab notebook.
- [ ] Test the code.
