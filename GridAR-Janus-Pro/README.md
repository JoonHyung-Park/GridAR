# GridAR-Janus-Pro

Baseline BoN reproduction and `GridAR` code on `T2I-CompBench++` with `Janus-Pro-7B`.

## Setup

Prepare `T2I-CompBench++` under `../benchmark/T2I-CompBench`:

```bash
cd ..
mkdir -p benchmark
cd benchmark
git clone https://github.com/Karine-Huang/T2I-CompBench.git
cd T2I-CompBench
git checkout 7728deac05e17903b3145befd9b7207c7c019573
```

Expected checkpoints:

- `../checkpoints/Janus_ckpt/Janus-Pro-7B` from `deepseek-ai/Janus-Pro-7B`
- `../checkpoints/Qwen2.5-VL-7B-Instruct` from `Qwen/Qwen2.5-VL-7B-Instruct`

## Installation

```bash
conda create -n gridar_janus python=3.10.12 -y
conda activate gridar_janus
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

`GridAR` additionally requires:

```bash
export OPENAI_API_KEY=...
```

## Configuration

- benchmark configs: [`configs/benchmark`](./configs/benchmark)
- method configs: [`configs/method`](./configs/method)
- model configs: [`configs/model_params`](./configs/model_params)

Default settings:

The default setup follows the main comparison setting reported in the paper tables: `GridAR` runs with `N=4`, while baseline `Best-of-N` runs with `N=8`.

## Running Experiments

BoN, one category:

```bash
bash scripts/run_bon_one.sh color
```

BoN, all categories:

```bash
bash scripts/run_bon_all.sh
```

GridAR, one category:

```bash
bash scripts/run_gridar_one.sh color
```

GridAR, all categories:

```bash
bash scripts/run_gridar_all.sh
```

To save intermediate stage outputs:

```bash
python3 evaluate_gridar.py save_intermediate_outputs=true ...
```

## Outputs

Images:

```text
../benchmark/T2I-CompBench/outputs/generated/Janus-Pro-7B/<run_name>/samples/
```

Records:

```text
../benchmark/T2I-CompBench/outputs/generated/Janus-Pro-7B/<run_name>/records/
```

Optional intermediate outputs:

```text
../benchmark/T2I-CompBench/outputs/generated/Janus-Pro-7B/<run_name>/intermediate_outputs/
```
