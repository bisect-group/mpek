# MPEK emulator_bench

This wrapper evaluates MPEK with its original settings on the default EMULaToR splits. It runs the `kcat` and `km` tasks in `data/processed/datasets/kinetic_dataset`. It runs three seeds (666, 777, 888) on every split folder it finds.

## Data

The default `--base_dir` is `<EMULaToR>/data/processed/datasets/kinetic_dataset`. The wrapper reads `<base_dir>/<task>/<split_group>[/threshold_<t>]/{train,val,test}.parquet`.

Split groups: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits`, `enzyme_structure_splits`, `substrate_splits`, `conformer_cosine_splits`. The threshold folders come from the disk. This gives 7 split folders per task, 42 runs for kcat and km with 3 seeds.

Columns used: `sequence`, `smiles`, `log10_value`. The wrapper does not use pH, temperature or organism. The unified datasets are not used.

The wrapper never writes under `data/`. It writes to `--output_dir` (default `<EMULaToR>/outputs/baselines/mpek`):

```text
embeddings/proteins_v3/, embeddings/ligands_v3/  cached ProtT5 and Mole-BERT vectors (v3 = cache version)
<task>/<split_group>/<split_name>/seed_<seed>/   metrics, predictions, checkpoint_last.pt
mpek_summary_runs.csv, mpek_summary.csv          written by aggregate_mpek_results.py
```

`<split_name>` is `threshold_<t>`, or `all` for a split group without threshold folders. The protein cache key holds the ProtT5 model id, the length cap (1024) and the exact input text. A new `CACHE_VERSION` in `common.py` builds new cache folders.

ProtT5 vectors that the CataPro cache already holds for the same input text are copied (the CataPro files stay unchanged). This covers all sequences of up to 1000 residues. The other vectors are computed with the original `ProteinEncoder`. Ligands that the original loader cannot featurize are listed in `ligand_cache_failures.json`. Their rows are dropped and counted as `rows_dropped` in `final_results_*.csv`.

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

## Speed

Batch size 1 is slow with the plain loop (about 345 steps/s on an RTX PRO 4500). Two exact speed-ups give about 3,200 steps/s, with the same losses, weights and predictions bit for bit (see `REPORT.md`):

- `ForeachChildTuningAdamW`: the same maths in multi-tensor kernels.
- CUDA graphs: every step (gather, forward, loss, backward, optimizer) is replayed from a graph. Predictions at batch size 1 also use graphs.

`--no_graph` selects the plain loop. The plain loop is also used on CPU, with `--amp`, and with `--batch_size` above 1.

## Multi-run path (`launch_multirun.py`)

`multirun_train.py` trains many independent batch-1 runs of one task in one process. It keeps the batch-1 maths of every run: own initial weights, own shuffled order (the CPU random stream of the single-run script), own dropout masks, own target scaling, one `ChildTuningAdamW` step per sample. A hand-written backward and one Triton update kernel replace autograd and the optimizer, and CUDA graphs replay the steps. The files per run are those of `train_single_split.py` (`OUT/<task>/<split_group>/<split_name>/seed_<s>/`, `checkpoint_last.pt` every epoch). A stopped group resumes from the checkpoints.

```bash
$PY emulator_bench/launch_multirun.py --skip_cache --tasks kcat --group_size 1   # all runs of a task, one after the other
```

`--group_size R` trains R runs together. On the RTX PRO 4500, R = 1 is the fastest (see `REPORT.md`): one run keeps its weights and Adam state in the 64 MB L2 cache, and R > 1 runs are limited by memory bandwidth. Do not start this launcher on run folders that another launcher is writing.

## GPUs and precision

`launch_parallel_retrain.py --gpus 0` runs one training at a time on GPU 0. More runs per GPU (`--runs_per_gpu`) are not faster: one run already fills the GPU. Use `--gpus 0 1` to use more GPUs. `run_split_benchmarks.py` runs one training at a time on `--device`. Both take `--amp` (mixed precision, bf16 on Ampere or newer, fp16 with GradScaler on older GPUs; plain loop). The default is the original fp32. `--dry_run` lists the runs.

## Tests

`verify_equivalence.sh` runs the equivalence tests against the original MPEK code on the CPU. `verify_equivalence.sh --gpu` adds the CUDA-graph and real-cache checks. See `REPORT.md`.

## Resuming

- Cache files that exist are skipped. Cache writes are atomic.
- Training writes `checkpoint_last.pt` every epoch. A restarted run resumes from it.
- A run with `final_results_test.csv` is skipped unless `--overwrite` is set.
