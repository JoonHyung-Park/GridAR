<h1 align="center"> Progress by Pieces: Test-Time Scaling for <br>Autoregressive Image Generation (CVPR 2026)
</h1>

[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb)](https://grid-ar.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.21185-b31b1b)](https://arxiv.org/abs/2511.21185)

<div align="center">
  <img src="assets/GridAR_figure_KOR.png" alt="GridAR" width="700"/>
</div>
<br>

<div align="center">
  <a href="https://joonhyung-park.github.io/" target="_blank">Joonhyung&nbsp;Park</a><sup>1*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://hyeongwon-jang.github.io/" target="_blank">Hyeongwon&nbsp;Jang</a><sup>1*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://linkedin.com/in/joowon-kim-8b7390262" target="_blank">Joowon&nbsp;Kim</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=UWO1mloAAAAJ&hl=ko&oi=ao" target="_blank">Eunho&nbsp;Yang</a><sup>1,2</sup>
  <br>
  <sup>1</sup> KAIST &emsp; <sup>2</sup>AITRICS &emsp;
  <br>
  <sup>*</sup>Equal Contribution &emsp; <br>
</div>
<br>

**GridAR** is a grid-structured test-time scaling framework for autoregressive image generation. By directing computation toward the most promising continuations, GridAR effectively expands the search space to elicit the best achievable outputs from visual AR models. With 4 candidates, GridAR outperforms Best-of-N (N=8), achieving +14.4% quality and -25.6% cost. Even when paired with an open-source verifier (MiniCPM-V 4.5, 8.7B), it delivers +11.7% gains at -27.6% lower cost.


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
