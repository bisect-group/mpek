# MPEK emulator_bench report

## Scope

- Wrapper: `baselines/mpek/emulator_bench`. Tasks: kcat and km (`kinetic_dataset/{kcat,km}`). No other task fits MPEK.
- The paper (camera_ready.tex L2178, L2456) uses MPEK only in a kcat assay-condition study. The ablation tables list MPEK for kcat and KM. This wrapper covers those two tasks with sequence and SMILES only (no pH, temperature, organism).
- `baselines/mpek/assay_emulator_bench` is a second wrapper for the assay-condition study. It was also cleaned (no Optuna, new paths, original settings). See `MPEK_assay_REPORT.md`. It reads the assay columns (`pH`, `Temperature`) that the kcat split files already hold.

## Method (MPEK, Briefings in Bioinformatics 2024)

- Inputs: enzyme sequence and substrate SMILES. pH, temperature and organism are optional.
- ProtT5-XL-U50 gives a 1024-d mean-pooled vector. Mole-BERT gives a 300-d mean-pooled vector. Both encoders are frozen.
- A linear layer maps the protein vector to 300-d. The two vectors are concatenated (600-d without conditions).
- A PLE multitask block (shared experts, task experts, gates) feeds one tower MLP per task. The paper trains kcat and Km together.
- Loss: MSE on train-normalised log10 targets.
- Paper: random 8:1:1 split, five repeats. Reports PCC 0.808 (kcat), 0.777 (Km).

## Original default settings

| Setting | Paper (p.9 "Model parameter") | Code `MTLKcatKM/pretrain.sh` | Used |
|---|---|---|---|
| Optimiser | Adam | `ChildTuningAdamW`, mode None (utils.py:145, train.py:313) | ChildTuningAdamW |
| Learning rate | 1e-4 | `HEAD_LR=1e-4` (pretrain.sh:55) | 1e-4, constant (train.py:319-320, no `--lr_warmup`) |
| Weight decay | not stated | 1e-6 (pretrain.sh:52) | 1e-6 |
| Betas, eps | not stated | (0.9, 0.999), eps 1e-6 (train.py:313, utils.py:151) | same |
| Batch size | 1 | `BATCH_SIZE=1` (pretrain.sh:17) | 1 |
| Epochs | 50 | `EPOCHS=100` (pretrain.sh:15) | 100 |
| Experts | 4 | `NUM_EXPERTS=4` (pretrain.sh:26) | 4 |
| Expert layers | 1 | `EXPERT_LAERYS=1` | 1 |
| Expert dim | 768 | `EXPERT_OUT=EXPERT_HIDDEN=768` | 768 |
| Expert dropout | 0.2 | `DROPOUT=0.3` (pretrain.sh:61) | 0.3 |
| PLE layers | not stated | 1 | 1 |
| Tower layers | 3 ("128, 128, or 1") | `TOWER_HID_LAYER=2` (pretrain.sh:46) | 2 |
| Tower hidden / dropout | 128 / not stated | 128 / `0 0` | 128 / 0 |
| Gradient clipping, LR schedule | not stated | none (train.py:76-77, 319-320) | none |
| Protein max length | not stated | 1024 | 1024 |
| Checkpoint rule | not stated | no validation selection; saves `checkpoint_<epoch>.pt` from epoch 25 (train.py:386) | model after last epoch |
| Precision | not stated | fp32 | fp32 |
| Conditions | pH, temperature, organism optional | pH and temperature on, organism off (pretrain.sh:107-108) | off (see Open issues) |

Paper and code disagree on epochs, dropout and tower depth. I used the code values, as the task requires. The released `evaluate.sh` differs again (5 experts, tower 2, dropout 0.2); it is the web-server inference setup, so I did not use it. `train.py` argparse defaults (lr 1e-5, 100 epochs, dropout 0) are overridden by `pretrain.sh`.

## What the wrapper does now

1. `cache_embeddings.py` finds all split jobs, takes the unique sequences and SMILES, and writes fp32 ProtT5 and Mole-BERT vectors to `<output_dir>/embeddings`.
2. `train_single_split.py` loads the cached vectors into memory, trains one split and one seed with the settings above, and logs train and validation metrics every epoch. It uses the model after the last epoch and writes train, validation and test metrics and predictions.
3. `run_split_benchmarks.py` (sequential) and `launch_parallel_retrain.py` (GPU slots) run every split job for seeds 666, 777, 888.
4. `aggregate_mpek_results.py` writes per-run and mean/std tables.
5. Results go to `--output_dir` (default `EMULaToR/outputs/baselines/MPEK`), never into `data/`.

Split jobs found on disk for each task: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits/threshold_*`, `substrate_splits/threshold_*`. Thresholds are discovered, not hardcoded. The old defaults `enzyme_structure_splits` and `conformer_cosine_splits` do not exist on disk and were removed.

## Files

Removed: `tune_optuna.py`, `launch_parallel_optuna.py`.

Changed: `common.py`, `dataset.py`, `modeling.py`, `train_single_split.py`, `cache_embeddings.py`, `prepare_assets.py`, `run_split_benchmarks.py`, `launch_parallel_retrain.py`, `aggregate_mpek_results.py`, `README.md`, `commands.txt`.

Removed from the code:
- Optuna scripts, text and the `best_hparams` hparam-json path.
- Hyperparameter flags in the orchestrators (settings now live only in `train_single_split.py` defaults).
- Cosine LR schedule, gradient clipping, validation-based best checkpoint, bf16/fp16 autocast, TF32, `--protein_dtype float16`, threshold-filter options, split-size columns.

## Bugs fixed

- Wrapper did not follow the original optimiser: it used `torch.optim.AdamW` (eps 1e-8, cosine decay, clip 1.0). It now uses the original `ChildTuningAdamW` with a constant lr.
- Wrapper selected the best validation epoch. The original has no such rule. The wrapper now uses the last epoch.
- Wrapper used 50 epochs, dropout 0.2, tower 3 layers (paper values). It now uses the code values.
- Cache used fp16 autocast and fp16 storage. It now runs ProtT5 in fp32 and stores fp32.
- Target std used a floor of 1e-8. The original uses `std + 1e-5`. Fixed.
- Results were written into the split folders under `data/`. They now go to `--output_dir`.
- Default paths pointed to `data/processed/baselines/MPEK` (does not exist) and listed split groups that do not exist. Fixed.
- `prepare_assets.py` tried to recover the Mole-BERT checkpoint from git history. The checkpoint is in the upstream HEAD, so this was replaced by a check with a clear message.
- Scripts imported `emulator_bench` without a path set-up when started as `python emulator_bench/x.py`. Each script now adds the repo root to `sys.path`.
- Parallel launcher: `--trials_per_gpu` alias and the `--default_settings` flag are removed.

## Deviations from the original protocol

- Cached frozen-encoder vectors (ProtT5, Mole-BERT). The encoders are frozen and in eval mode in the original, so the vectors do not change. The Mole-BERT path uses `GNN` plus `global_mean_pool`. This is meant to equal the frozen `GNN_graphpred` pooling, but it is unverified (see "Mole-BERT status"). ProtT5 uses the same tokenisation, `[UZOB]` to X, 1024 token limit, and mean over tokens without the end token.
- Features are held in memory and indexed by batch on the GPU (no `DataLoader`). Batch size stays 1.
- Shuffle and RNG pattern follow the original DataLoaders. Each epoch draws, from the global torch RNG, one iterator seed and one `RandomSampler` seed, then builds the permutation from a generator seeded with the sampler seed. The validation pass and the test pass of the original each draw one iterator seed, so the wrapper draws one seed before the validation pass and one for the test pass it does not run. With the same seed, the dropout masks and epoch order are the same as in the original on CPU.
- Validation and final prediction use the training batch size (`--batch_size`, default 1), in sequential order with a smaller last batch. The original evaluates at `args.batch_size` (`MTLKcatKM/pretrain.sh:17,66`; `train.py:280-291`). The PLE layer reshapes with `view(-1, num_experts, ...)` after `concat(dim=0)` (`layers/ple.py:73,79`). This mixes rows when the batch is larger than 1, so predictions depend on the batch size. The wrapper copies the original batch size and does not change it.
- The model has one task head (see "Why single-task").
- Loss and target normalisation follow the original: loss is sum of squares divided by (batch size + 1e-5), and the target mean and std are computed in float64 with `std + 1e-5`.
- Test metrics are computed once at the end. The original evaluates validation and test each epoch for logging only.
- The paper says bf16 for kinetic baselines (camera_ready.tex L2157). MPEK's original code uses fp32, so the wrapper uses fp32.

## Why single-task

The original MPEK model has two tasks (`num_tasks=2`). It trains kcat and Km together on one table with a masked loss, and the two tasks share the PLE experts.
The EMULaToR splits are per task: each task has its own train, val and test files, and the same enzyme-substrate pair can sit in different partitions for kcat and Km.
Training both tasks on one table would leak test pairs of one task into the training set of the other. So the wrapper trains one head per task, one run per task and split.
This changes the shared experts, which now learn from one task only. It is a deviation from the original model and should be labelled as such in the paper.

## Mole-BERT status

Unverified deviation. The `Mole-BERT` submodule is initialised at the pinned commit 2feff8a, and `Mole-BERT/model_gin/Mole-BERT.pth` is in it. The wrapper encodes ligands with upstream `GNN(5, 300, JK="last", drop_ratio=0.0, gnn_type="gin")` plus `global_mean_pool` (`cache_embeddings.py:load_molebert`).

- The original calls `GNN_graphpred(5, 300, drop_ratio=0.5)` (`MTLKcatKM/encoder/ligand_encoder.py:27`). Upstream `GNN_graphpred` cannot be built without `num_tasks`, so MPEK used a modified copy that we do not have.
- `train.py:204` always sets `frozen_params=True`. The encoder runs in eval mode under `no_grad` (`ligand_encoder.py:46-49`), so the dropout ratio has no effect and cached vectors are valid.
- `MODEL_CONFIG` in `ligand_encoder.py` sets `pool: "mean"`, which the wrapper copies.
- I did not compare the vectors with the fork's. Compare them if the fork becomes available.

## Changes after review

- `launch_parallel_retrain.py` and `run_split_benchmarks.py`: `--cache_device` now defaults to the first `--gpus` entry (launcher) or `--device` (sequential). Before, the cache step always used `cuda:0`.
- `cache_embeddings.py`: if some ligands fail to embed, the script writes `ligand_cache_failures.json`, prints the count and continues. Training drops only the rows of ligands listed there; any other missing embedding raises.
- Report text: the false statement about `assay_emulator_bench` is fixed (it was cleaned too). Added the single-task and Mole-BERT sections.
- Not changed: resume does not restore RNG state, so a resumed run is not bit-identical; pH and temperature off.
- Note: `--gpus` takes GPU ids. `--gpus 1` means GPU 1, not one GPU.
- Smoke after the changes: `launch_parallel_retrain.py --gpus 1 --skip_cache --tasks kcat --split_groups uniprot_time_splits --seeds 666 --epochs 1 --limit_rows 64` finished and wrote all metric files. The cache-device default and the failure message were checked by code only.

## Commands (all seeds x splits x tasks)

Run in `baselines/mpek` with `conda activate mldb`:

```bash
PY=python
$PY emulator_bench/prepare_assets.py
$PY emulator_bench/cache_embeddings.py --device cuda:0
$PY emulator_bench/launch_parallel_retrain.py --gpus 1 --runs_per_gpu 4
$PY emulator_bench/aggregate_mpek_results.py
```

Defaults: tasks `kcat km`, seeds `666 777 888`, five split groups. Add `--output_dir` and `--base_dir` to change folders. The sequential option is `run_split_benchmarks.py`.

## Smoke test

- Env: `/home/adhil/.conda/envs/mldb/bin/python` (torch 2.12, torch_geometric 2.8, transformers 5.19). All imports worked. No package was missing.
- The `Mole-BERT` submodule is initialised at the pinned commit 2feff8a, and the checkpoint is present. The first smoke runs used a scratch clone before this.
- Ran on GPU 1: `launch_parallel_retrain.py --tasks kcat --split_groups enzyme_sequence_splits uniprot_time_splits --seeds 666 777 --epochs 2 --limit_rows 64`. Cache (ProtT5 fp32, Mole-BERT) and 4 runs finished and wrote all metric files. Then `run_split_benchmarks.py` with seed 888 (1 epoch) and `aggregate_mpek_results.py` also worked. Resume and skip logic ran as designed on the repeat run.
- Nothing was written under `EMULaToR/data`.
- Speed: 113 steps/s at batch size 1 on a shared A10 GPU. One kcat run (55k rows, 100 epochs) takes about 13.6 h. A km run takes about 22 h. A full sweep is 15 runs per task. Use `--runs_per_gpu` (the GPU is mostly idle per run) and several GPUs.

## Open issues

- Choice made: code values (100 epochs, dropout 0.3, tower 2) over paper values (50, 0.2, 3). The camera-ready paper text and earlier MPEK numbers used the paper values. If the author wants the paper values, change the three defaults in `train_single_split.py`.
- Choice made: last-epoch model. The paper says validation-based checkpointing is kept "when the original has it". The original does not.
- Conditions off. The new kinetic parquet files have `pH`, `Temperature`, `Organism`. The original code enables pH and temperature in `pretrain.sh`. Camera-ready table (L2178) lists MPEK with sequence and SMILES only, so I kept them off. The assay-condition study is in `assay_emulator_bench`.
- Single-task heads instead of the original two-task model (see "Why single-task").
- The Mole-BERT embedding path is unverified (see "Mole-BERT status").
- Run time is long (see Smoke test). A full run needs several GPUs or long wall time.
- The `mpek.pdf` file is untracked in the submodule (known).
- `train_single_split.py` skips a finished run only when its `run_settings.json` matches `--epochs` and `--limit_rows`; otherwise it reruns. Ligands that cannot be featurized are listed in `ligand_cache_failures.json`. Training drops only their rows and reports the count as `rows_dropped` in `final_results_*.csv`.
