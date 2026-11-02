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

1. `cache_embeddings.py` finds all split jobs, takes the unique sequences and SMILES, and writes fp32 vectors to `<output_dir>/embeddings/{proteins_v3,ligands_v3}`.
   - ProtT5: the input text is the original one (`protein_text`). A vector is copied from the CataPro cache when CataPro embedded the same text (sequences of up to 1000 residues). The others are computed with the original `ProteinEncoder` class: padded to 1024, batch 1, fp32.
   - Mole-BERT: the graph comes from the original `mol_to_graph_data_obj_simple`. The frozen GNN runs in eval mode and the node vectors are mean-pooled. A ligand that raises in the loader is listed in `ligand_cache_failures.json` and its rows are dropped. Nothing is substituted.
2. `train_single_split.py` loads the cached vectors into memory, trains one split and one seed with the settings above, and logs train and validation metrics every epoch. It uses the model after the last epoch and writes train, validation and test metrics and predictions. On a CUDA device at batch size 1 it uses `fast_train.py` (see "Speed-ups").
3. `run_split_benchmarks.py` (sequential) and `launch_parallel_retrain.py` (GPU slots) run every split job for seeds 666, 777, 888. Both take `--dry_run`.
4. `aggregate_mpek_results.py` writes per-run and mean/std tables.
5. Results go to `--output_dir` (default `EMULaToR/outputs/baselines/mpek`), never into `data/`.

Split jobs found on disk for each task: `random_splits_grouped_sequence`, `random_splits_grouped_smiles`, `uniprot_time_splits`, `enzyme_sequence_splits/threshold_*`, `enzyme_structure_splits/threshold_*`, `substrate_splits/threshold_*`, `conformer_cosine_splits/threshold_*`. Thresholds are discovered, not hardcoded. This gives 7 jobs per task and 42 runs with 3 seeds (`launch_parallel_retrain.py --dry_run` lists them).

## Files

Added: `fast_train.py`, `verify_equivalence.sh`, `../tests/` (`test_original_equivalence.py`, `conftest.py`, `fixtures/`), `multirun_train.py`, `launch_multirun.py`, `../tests/test_multirun_equivalence.py` (multi-run path, see "Multi-run path").

Removed earlier: `tune_optuna.py`, `launch_parallel_optuna.py`.

Changed: `common.py`, `dataset.py`, `cache_embeddings.py`, `train_single_split.py`, `run_split_benchmarks.py`, `launch_parallel_retrain.py`, `README.md`, `commands.txt`, `REPORT.md`. Earlier changes: `modeling.py`, `prepare_assets.py`, `aggregate_mpek_results.py`.

No original file was edited.

## Speed-ups (all exact)

Measured on one RTX PRO 4500 Blackwell (cuda:0) with random fp32 features of the real shape (50,000 rows, 1024 + 300) and the real model and optimizer code. The CPU was shared with a UniKP job (load average 20-40), so CPU numbers are low. Step rate at batch size 1:

| Variant | steps/s | Same result as the plain loop |
|---|---|---|
| Plain loop, features on GPU (the wrapper before) | 345 | reference |
| Plain loop with `.item()` each step | 336 | yes |
| Foreach optimizer, eager | 538 | bit for bit (300 steps) |
| CUDA graph of the whole step, original optimizer | 2,300 | bit for bit |
| CUDA graph + foreach optimizer (used) | 3,184 | bit for bit (32,200 steps, test) |
| Same, Inductor-fused optimizer | 4,087 | no (loss diff 2.6e-6 after 200 steps); not used |
| CPU, 1 / 4 / 12 threads | 77 / 105 / 157 | yes |
| CPU, 12 processes x 1 thread (aggregate) | 347 | yes |
| GPU, N runs in one process on streams, N = 2 / 4 / 8 / 12 (aggregate) | 2,732 / 2,032 / 2,099 / 2,271 | yes |
| GPU, N processes, N = 2 / 4 / 8 (aggregate) | 2,973 / 3,005 / 2,997 | yes |

- One run on a real kcat split (50,289 train rows, 3 epochs, 2,795 validation rows each epoch): 73 s including the eager start. About 16 s per epoch afterwards.
- The graph holds 50 steps. The step size of ChildTuningAdamW is a host constant in a graph. In fp32 it has one value after step 15,900 (checked in a test). So the first 15,901 steps run eagerly and the rest run as graphs.
- Validation predictions at batch size 1 also run as graphs (bit for bit equal to the eager forward).
- Concurrency on one GPU does not help: the run is memory bound. Use one run at a time.
- Not tried: MPS, a bitwise-equal custom fused optimizer kernel (could give about 4,000 steps/s), `torch.compile` of the whole step.

### Multi-run path (`multirun_train.py`, `launch_multirun.py`)

This path trains R independent runs of one task in one process. Each run keeps the batch-1 maths of the single-run path: its own initial weights (`set_seed(seed)` and the model constructor), its own epoch orders (the CPU random stream of `train_single_split.py`, saved per run), one dropout mask per run and step, MSE on its own normalised target, and one `ChildTuningAdamW` step per sample.

- Parameters and Adam state of the R runs are stacked on a run axis. The forward pass uses batched matmuls. The backward pass is written by hand (checked against autograd). One Triton kernel applies `ChildTuningAdamW` to a parameter of all active runs from the rank-one gradient `d x^T`. It uses the op order and fused multiply-adds of the multi-tensor kernels and gives the same bits as `ForeachChildTuningAdamW` and `ChildTuningAdamW` (30 steps, all tensors, GPU test).
- The share gate feeds no output. As in the original it gets no gradient and no Adam state.
- Uneven train sizes: runs are sorted by size, largest first. At step j the active runs are the first k runs. A run that has finished its epoch is outside this prefix and no kernel touches it. Weights, Adam state and step count stay bit for bit (test). Every run does exactly 100 passes over its own rows. The step size comes from the step count of each run.
- Steps with a changing step size (the first 15,901 steps of a run) run eagerly. The rest run as 50-step CUDA graphs, as in `fast_train.py`. A graph run equals the eager run bit for bit (GPU test, 8,000 steps of two runs of different length).
- Dropout masks come from one device generator, reseeded to `seed * 1000 + epoch` of the first run at each epoch. The single-run path uses the global CUDA generator. Both draw independent Bernoulli(0.7) masks, so the two paths are not bit for bit equal when dropout is on. Tests inject the same masks into both paths.
- Evaluation per epoch uses one matmul per run and split, not a batch-1 loop. Rows do not depend on each other, so it equals the batch-1 pass up to float rounding.
- Output files, folder layout, `epoch_metrics.csv` and `checkpoint_last.pt` (same keys; the optimizer state is a real `ChildTuningAdamW` state) are those of the single-run path. A group resumes from the checkpoints at the epoch of the checkpoints. A single-run checkpoint also loads. If the runs of a group are at different epochs, the launcher stops and asks for `--overwrite`.

Speed. The GPU was shared with the production job (`mpek`, 100% utilisation) during all measurements. The numbers below are steps/s of that shared GPU. The same code ran next to the existing graph path (`fast_train.py`), which gave 1,480 steps/s in every measurement. The report measured 3,184 steps/s for it alone, so the share factor is about 2.15. The unshared column assumes the same factor for all variants (time slicing). It is an estimate (about +-25%). Re-measure when the GPU is free.

| Variant (random features, real model size) | steps/s aggregate, shared GPU | unshared estimate |
|---|---|---|
| Existing path: CUDA graph + foreach optimizer, 1 run | 1,480 | 3,184 (measured earlier) |
| Multi-run code, R = 1, eager | 2,810 | about 6,000 |
| **Multi-run code, R = 1, CUDA graphs** | **4,100** | **about 8,800** |
| R = 1, graphs, forward and backward with `torch.compile` (not used) | 4,720 | about 10,100 |
| R = 2 / 3 / 7 / 21, graphs | 2,570 / 2,780 / 2,740 / 2,720 | about 5,500 to 5,900 |
| 2 and 3 runs of R = 1 on separate streams | 2,830 / 2,760 | about 6,000 |
| End to end, R = 1, real kcat split (with evaluation and checkpoint each epoch, 13 s per epoch) | 4,020 | about 8,600 |
| End to end, R = 21, real kcat (7 splits x 3 seeds), 1 epoch with the 15,901 eager steps: 393 s | 2,690 | about 5,800 |

Why R = 1 is the fastest. A run holds weights, Adam m and v in 49 MB. That fits in the 64 MB L2 cache, so a step touches mostly L2. For R >= 2 the working set is larger than L2 and a step reads and writes about 130 to 160 MB per run from DRAM (24 B per parameter for Adam alone). The GPU is then limited by DRAM bandwidth and the rate no longer grows with R. The fused kernels (hand-written backward, one Triton update, graphs) are what give the gain over the existing path, not the number of runs per kernel. GPU memory at R = 21: 1.5 GB reserved for the stacked state and the graphs (measured), plus the train features (1.9 GB kcat, 2.6 GB km).

Projected wall clock for all 42 runs (254.8 M steps: 105.7 M kcat + 149.1 M km), unshared estimate:

| Layout | Time |
|---|---|
| Existing path (running now) | about 22 h train + 1 h evaluation |
| Multi-run code, R = 1 (`--group_size 1`, recommended) | about 8 h train + about 0.3 h evaluation and files |
| Multi-run code, R = 21 per task | about 12 h |

The remaining gap to "a few hours" is the work per step. Exact fp32 batch-1 training updates 4.1 M parameters with Adam (24 B per parameter read and written) at every step. The profile at R = 1 (shared GPU) shows 132 microseconds of kernel time per step, 64 of them in the update kernels and the rest in many small forward and backward kernels. Fusing these small kernels is the remaining lever (`torch.compile` gave +15%).

### Projected wall clock for all 42 runs (100 epochs, existing path)

Train rows: 352,316 (kcat, 7 splits) and 496,849 (km, 7 splits) per seed.

| Layout | Time |
|---|---|
| Exact batch 1, CUDA graphs + foreach, one run at a time (used) | 22.2 h train + about 1 h validation and final passes = about 23 h (kcat about 28 min and km about 38 min per run) |
| Original loop on the farm A10 (113 steps/s) | about 750 h |
| Batch 32 / 64 / 128 / 256 (not implemented, see below) | about 1.1 / 0.6 / 0.4 / 0.3 h of training |

### Batched variant (not implemented; deviation from the authors' method)

The original PLE layer reshapes with `torch.concat(outs, dim=0).view(-1, num_experts, expert_dim)` (`layers/ple.py:73,79`). At batch size above 1 this mixes rows between samples. The minimal fix is `torch.stack(outs, dim=1)`. It is identical at batch size 1. A wrapper can apply it without editing the original file.

Measured with the fix, CUDA graphs and the foreach optimizer (steps/s, epoch of 50k rows, 100 epochs): batch 32: 2,093, 0.75 s, 1.2 min. Batch 64: 1,910, 0.4 s, 0.7 min. Batch 128: 1,438, 0.3 s, 0.5 min. Batch 256: 1,065, 0.2 s, 0.3 min.

Each epoch makes B times fewer updates, and lr 1e-4 and 100 epochs were set for batch size 1. I did not test the accuracy with a larger batch. Any change of the batch size, lr or epochs is a deviation from the authors' method and needs a decision.

## Bugs fixed

- Wrapper did not follow the original optimizer: it used `torch.optim.AdamW` (eps 1e-8, cosine decay, clip 1.0). It now uses the original `ChildTuningAdamW` with a constant lr.
- Wrapper selected the best validation epoch. The original has no such rule. The wrapper now uses the last epoch.
- Wrapper used 50 epochs, dropout 0.2, tower 3 layers (paper values). It now uses the code values.
- Cache used fp16 autocast and fp16 storage. It now runs ProtT5 in fp32 and stores fp32.
- Target std used a floor of 1e-8. The original uses `std + 1e-5`. Fixed. Labels are now cast to float32 before the normalisation, as the original does (mean and std stay float64).
- Results were written into the split folders under `data/`. They now go to `--output_dir`.
- Default paths pointed to `data/processed/baselines/MPEK` (does not exist). Fixed.
- The default split groups missed `enzyme_structure_splits` and `conformer_cosine_splits`, which exist on disk (one threshold each). Now 7 groups and 42 runs.
- The cache changed the sequence text (upper case, no spaces, `U/Z/O/B`) and re-implemented the Mole-BERT graph builder. It now uses the original text rule and the original `mol_to_graph_data_obj_simple`.
- The old cache wrote one `.npz` per vector and used the model id as the only protein key. The new cache (v3) uses the exact input text in the key and one `.npy` per vector.
- `prepare_assets.py` tried to recover the Mole-BERT checkpoint from git history. It now checks that the checkpoint exists.
- Scripts imported `emulator_bench` without a path set-up when started as `python emulator_bench/x.py`. Each script now adds the repo root to `sys.path`.
- Parallel launcher: `--trials_per_gpu` alias and the `--default_settings` flag are removed.
- The orchestrators now take `--dry_run`.

## Deviations from the original protocol

- Cached frozen-encoder vectors. The encoders are frozen and in eval mode in the original, so the vectors do not change. ProtT5: same tokenizer settings (`legacy=False`), `[UZOB]` to X, 1024-token limit, mean over tokens without the end token. A vector copied from CataPro came from the same text and the same pooling, but from a batched, padded-to-longest run with another tokenizer object. Token ids are identical for all 29,344 copied sequences and the vectors differ from the original encoder by at most 1.7e-7 (float rounding).
- Mole-BERT: upstream `GNN` plus `global_mean_pool` in eval mode on batches of molecules, against the original one molecule per batch. Differences are float rounding. See "Mole-BERT status" for the part that cannot be checked.
- Features are held in memory and indexed on the GPU (no `DataLoader`). Batch size stays 1.
- Shuffle and RNG pattern follow the original DataLoaders. Each epoch draws, from the global torch RNG, one iterator seed and one `RandomSampler` seed, then builds the permutation from a generator seeded with the sampler seed. The validation pass and the test pass of the original each draw one iterator seed, so the wrapper draws one seed before the validation pass and one for the test pass it does not run. With the same seed, the dropout masks and epoch order are the same as in the original on CPU (checked against the original `train` and `evaluate`).
- Validation and final prediction use the training batch size (`--batch_size`, default 1), in sequential order. The original evaluates at `args.batch_size` (`MTLKcatKM/pretrain.sh:17,66`; `train.py:280-291`). Predictions depend on the batch size because of the PLE reshape (see above). The wrapper copies the original batch size and does not change it.
- The model has one task head (see "Why single-task").
- Loss and target normalisation follow the original: loss is sum of squares divided by (batch size + 1e-5), and the target mean and std are computed in float64 with `std + 1e-5`.
- Test metrics are computed once at the end. The original evaluates validation and test each epoch for logging only.
- The paper says bf16 for kinetic baselines (camera_ready.tex L2157). MPEK's original code uses fp32, so the wrapper uses fp32.
- The original `train.py` sets no seed. The wrapper seeds Python, NumPy and torch with the run seed. The head weights equal the original model's head weights for the same seed (checked).
- Resume restores weights and optimizer state, not the RNG state. A resumed run is not bit-identical.

## Why single-task

The original MPEK model has two tasks (`num_tasks=2`). It trains kcat and Km together on one table with a masked loss, and the two tasks share the PLE experts.
The EMULaToR splits are per task: each task has its own train, val and test files, and the same enzyme-substrate pair can sit in different partitions for kcat and Km.
Training both tasks on one table would leak test pairs of one task into the training set of the other. So the wrapper trains one head per task, one run per task and split.
This changes the shared experts, which now learn from one task only. It is a deviation from the original model and should be labelled as such in the paper.

## Mole-BERT status

- The `Mole-BERT` folder holds `model_gin/Mole-BERT.pth` (md5 equal to the farm copy). The wrapper encodes ligands with upstream `GNN(5, 300, JK="last", drop_ratio=0.0, gnn_type="gin")` plus `global_mean_pool`.
- The original calls `GNN_graphpred(5, 300, drop_ratio=0.5)` (`MTLKcatKM/encoder/ligand_encoder.py:27`) from a package named `Mole_BERT`. Upstream `GNN_graphpred` cannot be built without `num_tasks`, so MPEK used a modified copy that we do not have. The wrapper registers the folder as `Mole_BERT` in `sys.modules` (`common.py`).
- `train.py:204` always sets `frozen_params=True`. The encoder runs in eval mode under `no_grad` (`ligand_encoder.py:46-49`), so the dropout ratio has no effect and cached vectors are valid. `MODEL_CONFIG` in `ligand_encoder.py` sets `pool: "mean"` and `drop_ratio` 0.0.
- Check: the test `test_molebert_vectors_equal_original_encoder` runs the original `LigandEncoderMoleBert` class on a reconstruction of the copy (upstream `GNN_graphpred` with `num_tasks=1`, `from_pretrained(path, device)`, forward returning the mean-pooled node vectors). The wrapper vectors equal it within float rounding on the fixture molecules (max abs diff in the test output).
- Graphs: the original `mol_to_graph_data_obj_simple` and the original `MolTestDataset.__getitem__` run without error on all 13,104 unique SMILES of the 14 split jobs. No ligand is dropped.
- Verdict: EQUIVALENT-EXCEPT-DOCUMENTED. The only unverified part is that the reconstruction equals the authors' copy. Everything in the released code points to it (embedding size 300, `pool: "mean"`, frozen eval mode, `JK="last"` default). Compare with the fork if it becomes available.

## Commands (all seeds x splits x tasks)

See `commands.txt`. In short, from `baselines/mpek` with the `mldb` environment:

```bash
PY=/home/da24s023/miniforge3/envs/mldb/bin/python
$PY emulator_bench/prepare_assets.py
$PY emulator_bench/cache_embeddings.py --device cuda:0
$PY emulator_bench/launch_parallel_retrain.py --gpus 0 --dry_run          # lists the 42 runs
$PY emulator_bench/launch_parallel_retrain.py --gpus 0 --skip_cache --no_progress
$PY emulator_bench/aggregate_mpek_results.py
```

Defaults: tasks `kcat km`, seeds `666 777 888`, seven split groups. Add `--output_dir` and `--base_dir` to change folders.

## Caches (built on 2026-10-29)

`outputs/baselines/mpek/embeddings`: 30,268 ProtT5 vectors (29,344 copied from CataPro, 924 computed in about 3 min on the RTX PRO 4500: 121 sequences of 1001-1023 residues, where CataPro cut 500+500 and MPEK does not, and 803 of more than 1023 residues, which MPEK truncates to 1023 residues plus the end token) and 13,104 Mole-BERT vectors. 0 ligands failed, so 0 rows are dropped in every split.

## Smoke test

All outputs are in the session scratchpad, not in the repository.

- Tiny end to end: `launch_parallel_retrain.py --gpus 0 --tasks kcat --split_groups uniprot_time_splits enzyme_structure_splits --seeds 666 --epochs 2 --limit_rows 64` built a cache (87 sequences: 86 copied from CataPro, 1 computed; 203 SMILES, 0 failed), ran 2 runs and `aggregate_mpek_results.py` wrote both tables.
- `launch_parallel_retrain.py --gpus 0 --dry_run` lists 42 runs (7 split folders x 2 tasks x 3 seeds).
- Real size: kcat `random_splits_grouped_sequence`, seed 666, 3 epochs with the real caches: 73 s (eager start included). Train RMSE 1.35, 1.22, 1.16 and validation RMSE 1.28, 1.19, 1.12 (log10 units) over the epochs.
- Resume: a run killed after epoch 1 resumed from the checkpoint and finished (the RNG is not restored, so later epochs differ from an uninterrupted run).
- Nothing was written under `EMULaToR/data`. The CataPro cache was only read.

## Equivalence verdict

Run `emulator_bench/verify_equivalence.sh` (CPU) and `verify_equivalence.sh --gpu`. Last run: 13 CPU tests passed with 0 skipped, and 5 GPU tests passed. The tests call the original code (`MTLModel`, `train`, `evaluate`, `ChildTuningAdamW`, `TrainNormalizer`, `ProteinEncoder`, `MTLKDataset.__getitem__`, `LigandEncoderMoleBert`, `mol_to_graph_data_obj_simple`, `MolTestDataset`).

| Check (order as in the spec) | Result |
|---|---|
| Features: ProtT5 input text, token ids and mask (U/Z/O/B, 1024 and 64 truncation, padding) | identical to `MTLKDataset.__getitem__` |
| Features: ProtT5 vectors (tiny random T5 on CPU) | identical to the original `ProteinEncoder`, bit for bit |
| Features: ProtT5 cache with the real model (27 sequences, including 1001-1023 and >1023 residues) | max abs diff 1.73e-7 against the original encoder (padded to 1024, batch 1) |
| Features: CataPro token ids for the 29,344 reused sequences | 0 differ |
| Features: Mole-BERT vectors (55 fixture molecules, real weights) | max abs diff 8.6e-7 against the original `LigandEncoderMoleBert` on the reconstructed copy |
| Features: graphs of all 13,104 real SMILES | original loaders raise on none; 0 rows dropped |
| Features: targets | identical to `TrainNormalizer` + float32 labels |
| Initial weights after seeding | all 26 tensors equal to the original `MTLModel` head |
| Optimizer, loss, batch order, dropout (3 epochs, 72 steps, CPU) | per-step losses, weights and validation predictions bit for bit equal to the original `train` and `evaluate` |
| Foreach optimizer vs `ChildTuningAdamW` (CPU and GPU) | bit for bit |
| CUDA graphs vs the plain loop (GPU, 32,200 steps, with the switch from eager to graph) | train predictions, eval predictions, weights and Adam state bit for bit |
| Real kcat split, 2 epochs, graph run vs plain run | `predictions_{train,val,test}.csv` and `epoch_metrics.csv` byte-identical |
| Multi-run path: hand-written backward vs autograd (per tensor) | max error to an fp64 reference about 4e-8 to 7e-6 of the largest gradient, the same size as the fp32 autograd error |
| Multi-run path: init, epoch orders | initial weights and the per-run order and CPU random stream equal the single-run script, also when runs are interleaved |
| Multi-run vs single run, same init, orders and injected dropout masks, 3 runs, 2,000 steps, CPU small model | weights rel 3.3e-7, Adam state rel 2.3e-6, per-step predictions max abs 3.7e-7 |
| Same, full model on GPU, 3 runs | after 1,200 steps: weights rel 5.2e-6 (abs 4.4e-7). After 2,000 steps: per-step predictions max abs 7.0e-5, eval predictions max abs 1.0e-4, weights abs 1.0e-3 (rel 1.8e-2 of the largest weight). Rounding grows with the step count: a 1e-7 relative change of the initial weights in the single-run path grows the same way (see below) |
| Uneven sizes (60, 45, 30 rows, 3 epochs, CPU) | weights rel 2.1e-7, Adam state rel 7.1e-7 against single runs of their own length |
| Run outside the active prefix | weights, m and v bit for bit unchanged |
| Triton update vs `ForeachChildTuningAdamW` and `ChildTuningAdamW` | bit for bit (30 steps, all tensors) |
| CUDA graphs vs eager steps of the multi-run path | bit for bit (8,000 steps, 2 runs of different length) |
| Real kcat split (`random_splits_grouped_sequence`), 3 seeds, 3 epochs, normal dropout | validation RMSE of the group vs single runs, gap of the 3-seed means 0.0085 / 0.0015 / 0.0070 for epochs 1 / 2 / 3; seed spread of the single runs 0.031 / 0.006 / 0.031 |
| Concurrency | no wrapper code: runs are separate processes. No speed gain, so the default is 1 run per GPU |

Verdict: EQUIVALENT-EXCEPT-DOCUMENTED. The documented deviations are the single-task head, no pH, temperature or organism input, cached frozen encoders (differences at float rounding), the last-epoch model, and the reconstructed Mole-BERT copy. The speed-ups do not change results.

Rounding control for the 2,000-step row: the single-run loop (full model, GPU) was run with initial weights changed by a relative 1e-7 noise. For three seeds its predictions moved by max abs 1.6e-4, 1.3e-2 and 1.1e-6 after 2,000 steps (3.5e-7, 3.9e-7, 3.0e-7 after 1,000). The multi-run path differed from the unperturbed run by 1.4e-6, 3.7e-2 and 6.2e-3 after 2,000 steps (4.8e-7, 3.3e-7, 4.8e-7 after 1,000). Memorising random targets amplifies rounding differences after about 1,500 steps in both paths. This is not a defect of the multi-run path. The multi-run verdict is EQUIVALENT up to float rounding; the downstream check shows no difference between the multi-run and the single-run validation curves beyond the seed spread.

## Open issues

- The released code does not run as shipped (the wrapper follows `pretrain.sh` values and `train.py` logic, and the original code is not edited): `layers/__init__.py` has no `MMoE`, so `model.py` and `train.py` do not import; `GNN_graphpred` needs the authors' copy; `train.main` passes `normalizer=` to `evaluate`, which does not take it; `pretrain.sh` passes flags the `train.py` parser does not define and has unfilled placeholders; `train.py` sets no seed. The tests add shims for the first two and call `train` and `evaluate` directly.
- Choice made: code values (100 epochs, dropout 0.3, tower 2) over paper values (50, 0.2, 3). The camera-ready paper text and earlier MPEK numbers used the paper values. If the author wants the paper values, change the three defaults in `train_single_split.py`.
- Choice made: last-epoch model. The paper says validation-based checkpointing is kept "when the original has it". The original does not.
- Conditions off. The new kinetic parquet files have `pH`, `Temperature`, `Organism`. The original code enables pH and temperature in `pretrain.sh`. Camera-ready table (L2178) lists MPEK with sequence and SMILES only, so I kept them off. The assay-condition study is in `assay_emulator_bench`.
- Single-task heads instead of the original two-task model (see "Why single-task").
- The Mole-BERT encoder is checked against a reconstruction of the authors' copy, not the copy itself.
- Batch size 1 as in the code: about 23 h for the 42 runs on one GPU. A larger batch is much faster but is a deviation that needs a decision on the PLE fix, lr and epochs.
- The `mpek.pdf` file is untracked in the submodule (known).
- `train_single_split.py` skips a finished run only when its `run_settings.json` matches `--epochs` and `--limit_rows`; otherwise it reruns. Ligands that cannot be featurized are listed in `ligand_cache_failures.json`. Training drops only their rows and reports the count as `rows_dropped` in `final_results_*.csv`.
- Multi-run path (not used by the running job): the unshared speed is an estimate from a shared GPU. Dropout masks come from another random stream than the single-run path, so runs are not bit for bit equal to single runs with dropout on. A group checkpoint is written run by run, so a crash during the writes can leave runs at different epochs (the launcher stops and asks for `--overwrite`). `torch.compile` of the forward and backward gave +15% at R = 1 and was not used. A fused forward and backward kernel could give more.
- The CataPro ProtT5 cache is read only. If its files change, rebuild with `cache_embeddings.py --overwrite`.
