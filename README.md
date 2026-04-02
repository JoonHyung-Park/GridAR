# Progress by Pieces: Test-Time Scaling for Autoregressive Image Generation

CVPR 2026

Joonhyung Park*, Hyeongwon Jang*, Joowon Kim, Eunho Yang (* Equal contribution)

KAIST, AITRICS

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb)](https://grid-ar.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.21185-b31b1b)](https://arxiv.org/abs/2511.21185)

## Introduction

This repository contains the `GridAR` codebase and is organized around two model-specific subprojects:

- [`GridAR-Janus-Pro`](./GridAR-Janus-Pro): Janus-Pro based `GridAR` and `BoN` evaluation on `T2I-CompBench++`
- [`GridAR-EditAR`](./GridAR-EditAR): EditAR-based editing evaluation on `PIE-bench`

For integration-specific details on each backbone and method combination, refer to the `README.md` inside each subproject directory.

###  Layout

```text
GridAR/
├── GridAR-Janus-Pro/
├── GridAR-EditAR/
├── checkpoints/
│   ├── Janus_ckpt/
│   │   └── Janus-Pro-7B/
│   ├── Qwen2.5-VL-7B-Instruct/
│   └── EditAR_ckpt/
├── benchmark/
│   ├── T2I-CompBench/
│   └── PIE-bench/
```

## Citation

```bibtex
@article{park2025progress,
  title={Progress by Pieces: Test-Time Scaling for Autoregressive Image Generation},
  author={Park, Joonhyung and Jang, Hyeongwon and Kim, Joowon and Yang, Eunho},
  journal={arXiv preprint arXiv:2511.21185},
  year={2025}
}
```

CVPR BibTeX will be added later.
