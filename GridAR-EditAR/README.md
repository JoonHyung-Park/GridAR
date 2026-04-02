# GridAR-EditAR

Baseline `EditAR` BoN reproduction and `GridAR-EditAR` code for `PIE-Bench`.

## Installation

```bash
conda create -n gridar_editar python=3.10.12 -y
conda activate gridar_editar
pip install -r requirements.txt
```

## Checkpoints

Expected paths under `GridAR/`:

- `../checkpoints/EditAR_ckpt/editar_release.pt`
- `../checkpoints/EditAR_ckpt/pretrained_models/vq_ds16_t2i.pt`
- `../checkpoints/EditAR_ckpt/pretrained_models/t5-ckpt/flan-t5-xl/`
- `../checkpoints/Qwen2.5-VL-7B-Instruct` from `Qwen/Qwen2.5-VL-7B-Instruct`

For the original `EditAR` assets under `../checkpoints/EditAR_ckpt/`, follow the upstream checkpoint preparation guide:
https://github.com/JitengMu/EditAR

## Running Experiments

Baseline EditAR BoN:

```bash
bash scripts/run_bon_edit_pie.sh
python3 evaluation/evaluation.py --test_name BoN__EditAR__PIE__N4
```

GridAR main method:

```bash
export OPENAI_API_KEY=...
bash scripts/run_gridar_edit_pie.sh
python3 evaluation/evaluation.py --test_name GridAR__EditAR__PIE__N4
```

Note: To reduce unnecessary API cost, the EditAR setting runs prompt refinement once rather than at both intermediate stages.

## Outputs

Generated results:

```text
../benchmark/PIE-Bench/outputs/<run_name>/visualization/
```

Verifier-selected targets:

```text
../benchmark/PIE-Bench/outputs/<run_name>/visualization/*_tgt_best.png
```

Intermediate selection and refine records:

```text
../benchmark/PIE-Bench/outputs/<run_name>/records/
```
