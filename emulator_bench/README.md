# MPEK emulator_bench

This wrapper evaluates MPEK with its original settings on the default EMULaToR splits. It runs the `kcat` and `km` tasks in `data/processed/datasets/kinetic_dataset`. It runs three seeds (666, 777, 888) on every split folder it finds.

## Data

The default `--base_dir` is `<EMULaToR>/data/processed/datasets/kinetic_dataset`. The wrapper reads `<base_dir>/<task>/<split_group>[/threshold_<t>]/{train,val,test}.parquet`.

Split groups: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits`, `substrate_splits`. The threshold folders come from the disk.

Columns used: `sequence`, `smiles`, `log10_value`. The wrapper does not use pH, temperature or organism. The unified datasets are not used.

The wrapper never writes under `data/`. It writes to `--output_dir` (default `<EMULaToR>/outputs/baselines/MPEK`):

```text
embeddings/proteins_v2/, embeddings/ligands_v2/  cached ProtT5 and Mole-BERT vectors (v2 = cache version)
<task>/<split_group>/<split_name>/seed_<seed>/   metrics, predictions, checkpoint_last.pt
mpek_summary_runs.csv, mpek_summary.csv          written by aggregate_mpek_results.py
```

`<split_name>` is `threshold_<t>`, or `all` for a split group without threshold folders. The protein cache key holds the ProtT5 model id and the length cap (1024). A new `CACHE_VERSION` in `common.py` builds new cache folders.

## Method and settings

MPEK concatenates a frozen ProtT5 vector (1024-d, projected to 300-d) with a frozen Mole-BERT vector (300-d). A PLE block of expert layers and gates and a tower MLP predict the value. The wrapper caches the frozen vectors in fp32. Each task trains alone because each task has its own splits.

Settings follow the released training script `MTLKcatKM/pretrain.sh` and `MTLKcatKM/utils.py`:

- `ChildTuningAdamW` (mode off), lr 1e-4, weight decay 1e-6, betas (0.9, 0.999), constant lr, no gradient clipping.
- Batch size 1, 100 epochs, masked MSE on train-normalised targets, fp32.
- 4 experts, expert layers 1, expert dim 768, expert dropout 0.3, 1 PLE layer.
- Tower: 2 layers, hidden 128, dropout 0.
- ProtT5 input length 1024. The wrapper takes the model after the last epoch. The original code has no checkpoint selection.

`REPORT.md` lists the differences between the paper and the code.

## Setup

The Mole-BERT submodule must be present:

```bash
git submodule update --init Mole-BERT
```

Run all commands from the MPEK repository root in the `mldb` conda environment. `commands.txt` has the commands.

## GPUs and precision

`launch_parallel_retrain.py --gpus 0 1 --runs_per_gpu 2` runs four trainings at once. Each run gets one GPU through `CUDA_VISIBLE_DEVICES`. `run_split_benchmarks.py` runs one training at a time on `--device`. Both take `--amp`: mixed precision (bf16 on Ampere or newer, fp16 with GradScaler on older GPUs). The default is the original fp32.

## Resuming

- Cache files that exist are skipped. Cache writes are atomic.
- Training writes `checkpoint_last.pt` every epoch. A restarted run resumes from it.
- A run with `final_results_test.csv` is skipped unless `--overwrite` is set.
