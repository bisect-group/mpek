# MPEK assay_emulator_bench

This wrapper runs the kcat assay-condition study of the Leak-CURBER paper. It trains MPEK on the EMULaToR kcat splits with and without the original pH and temperature inputs. It uses the original MPEK settings and three seeds (666, 777, 888).

## Data

The default `--base_dir` is `<EMULaToR>/data/processed/datasets/kinetic_dataset`. The wrapper reads `kcat/<split_group>[/threshold_<t>]/{train,val,test}.parquet`.

Default split groups: `enzyme_sequence_splits` (P-seq) and `uniprot_time_splits` (Time). Pass `--split_groups` to add `random_splits_grouped_sequence`, `random_splits_grouped_smiles` or `substrate_splits`. The threshold folders come from the disk. The unified datasets are not used.

Columns used:

- `sequence`, `smiles`, `log10_value`: input and target.
- `pH`: used as is. A missing value becomes 7.0.
- `Temperature` (kelvin): converted to Celsius, as the original MPEK encoder expects. A missing value becomes 298.15 K.
- `Organism` exists but stays off, as in the paper study.

The kcat files hold one row per assay. One (sequence, smiles) pair can have several rows with different pH and temperature. All rows of a pair stay in the same partition.

## Method and settings

Both modes use the same rows and the same settings as `emulator_bench`: `ChildTuningAdamW`, lr 1e-4, weight decay 1e-6, batch size 1, 100 epochs, 4 experts, expert dim 768, expert dropout 0.3, tower 2 layers of 128, fp32, model after the last epoch. `emulator_bench/REPORT.md` lists the sources and the paper differences.

- `with_conditions`: the original `AuxiliaryEncoder` (RBF banks for pH and temperature, 300-d) is concatenated with the ligand and projected protein vectors (900-d).
- `without_conditions`: protein and ligand only (600-d), on the same assay rows.

Evaluation: the wrapper takes the maximum true and the maximum predicted log10 value over the assays of each pair. It writes pair-level metrics (`pair_*` columns) and assay-row metrics for train, val and test.

## Setup

Run `emulator_bench/prepare_assets.py` and `emulator_bench/cache_embeddings.py` first. The wrapper reads the same versioned embedding cache (default `<EMULaToR>/outputs/baselines/MPEK/embeddings`, with `proteins_v<N>` and `ligands_v<N>` inside). The kcat sequences and SMILES are the same files, so the cache covers them. Run all commands from the MPEK repository root in the `mldb` conda environment. `commands.txt` has the commands.

Output (default `--output_dir` is `<EMULaToR>/outputs/baselines/MPEK_assay`):

```text
<mode>/kcat/<split_group>/<split_name>/seed_<seed>/   metrics, predictions, checkpoint_last.pt
mpek_assay_summary_runs.csv, mpek_assay_summary.csv
```

## GPUs and precision

`--gpus 0 1 --runs_per_gpu 2` runs four trainings at once. Each run gets one GPU through `CUDA_VISIBLE_DEVICES`. `--amp` turns on mixed precision (bf16 on Ampere or newer, fp16 with GradScaler on older GPUs). The default is the original fp32.

## Resuming

- Training writes `checkpoint_last.pt` every epoch. A restarted run resumes from it.
- A run with `final_results_test.csv` is skipped unless `--overwrite` is set.
