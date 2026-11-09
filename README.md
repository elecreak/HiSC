# HiSC: Hierarchical Spatial Clustering Token Compression for Efficient 3D Scene Understanding

Official inference implementation of **HiSC**. HiSC compresses spatially grounded visual tokens in [Video-3D LLM](https://github.com/LaVi-Lab/Video-3D-LLM) without training or changing its released checkpoint.

HiSC has two stages. Spatial graph merging groups visual patches that are close in 3D and semantically similar before language-model inference. Hierarchical spatial clustering then allocates a token budget to object-centric clusters and prunes tokens at selected Qwen2 layers, retaining spatial coverage and useful detail. The implementation covers ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, and SQA3D.

## Model and data

Use the [released Video-3D LLM Qwen Uniform-32 checkpoint](https://huggingface.co/zd11024/Video3D-LLM-LLaVA-Qwen-Uniform-32) and its [processed data](https://huggingface.co/datasets/zd11024/Video-3D-LLM_data). Follow the upstream [data preparation guide](scripts/3d/preprocessing/README.md) for the expected RGB, depth, camera, and annotation layout. No HiSC weights or fine-tuning are required.

## Installation

```bash
git clone https://github.com/elecreak/HiSC.git
cd HiSC
conda create -n hisc python=3.10 -y
conda activate hisc
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

The code follows the upstream Video-3D LLM dependency versions. It uses SciPy connected components; SciPy is included in the training extras. FlashAttention 2 is used by the released checkpoint evaluation path.

## Evaluation

From the repository root, set the checkpoint and processed-data locations and run all five validation tasks:

```bash
export CKPT_PATH=/path/to/Video3D-LLM-LLaVA-Qwen-Uniform-32
export DATA_ROOT=/path/to/Video-3D-LLM_data
export GPU_IDS=0,1
export N_GPU=2
bash scripts/3d/eval/eval_hisc.sh
```

## Acknowledgements and citation

This repository is based on [Video-3D LLM](https://github.com/LaVi-Lab/Video-3D-LLM), which builds on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).

```bibtex
@inproceedings{qu2026hisc,
  title={HiSC: Hierarchical Spatial Clustering Token Compression for Efficient 3D Scene Understanding},
  author={Qu, Jiuhe and Liang, Yingping and Fu, Ying},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2026}
}
```

The upstream license is [Apache-2.0](LICENSE).
