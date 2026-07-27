# MPEK EMULaToR Bench Plan

## 1. Baseline Summary From README

- Baseline repository: `/home/adhil/github/EMULaToR/baselines/mpek`
- Task: MPEK predicts enzymatic kinetic parameters from an enzyme sequence and substrate SMILES, with optional organism, pH, and temperature inputs.
- Documentation inspected: root `README.md`, native training/evaluation code, and `assay_emulator_bench/README.md`.
- Native condition path: pH and temperature pass through separate radial-basis-function banks, project to 300 dimensions, sum, and join the 300-dimensional ligand and projected-protein vectors.
- Training and evaluation: the wrapper caches native ProtT5 and Mole-BERT representations, trains through `assay_emulator_bench/train_single_split.py`, and evaluates within the same command.
- Defaults: AdamW, batch size 1, 50 epochs, learning rate `1e-4`, four experts, and the existing PLE/tower dimensions remain unchanged.
- Code mismatch: importing `MTLKcatKM.encoder` also imports the uninitialized Mole-BERT submodule. The wrapper will load the native `auxiliary_encoder.py` file directly and will not edit upstream MPEK code.

## 2. Expected Input Format

- Files: train, validation, and test Parquet files.
- Required fields: `sequence`, `smiles`, and `log10_value`.
- Assay-aware fields: finite `pH` and `Temperature`, with temperature stored in kelvin in the expanded dataset.
- Pair identifier: exact `(sequence, smiles)`.
- Provenance fields retained when present: `assay_mapping_status`, `pH_imputed`, and `Temperature_imputed`.
- Pair membership must remain disjoint across train, validation, and test.

## 3. Featurization and Preprocessing Path

- Protein features: cached native ProtT5 1024-dimensional mean-pooled embeddings.
- Ligand features: cached native Mole-BERT 300-dimensional mean-pooled embeddings.
- Cache root: `/home/adhil/github/EMULaToR/data/processed/baselines/MPEK/embeddings`.
- The cache is complete for all 19,171 sequences and 10,153 substrates in the assay-expanded data.
- No assay feature cache is required. The trainable native RBF encoder consumes pH and temperature in each batch.
- Kelvin values convert to Celsius before encoding.
- Invalid pH values outside `[0,14)` become `7.0`. Temperatures below 0 or above 100 degrees Celsius become `25.0`. Exactly 100 degrees Celsius remains unchanged.

## 4. Training and Evaluation Entrypoints

- Single run: `assay_emulator_bench/train_single_split.py`.
- Sequential orchestration: `assay_emulator_bench/run_split_benchmarks.py`.
- Parallel orchestration: `assay_emulator_bench/launch_parallel_retrain.py`.
- Optional tuning: `assay_emulator_bench/tune_optuna.py` and `assay_emulator_bench/launch_parallel_optuna.py`.
- No-assay mode keeps the existing 600-dimensional PLE input.
- Assay-aware mode uses the native 300-dimensional condition vector and a 900-dimensional PLE input.
- Training uses assay-row MSE and row-level target normalization.
- Validation records assay-row and pair-level metrics. Maximum true and predicted log10 values give one result per pair. Pair-level validation RMSE selects the best checkpoint.
- `checkpoint_latest.pt` supports resume. `checkpoint_best.pt` contains the pair-selected model.

## 5. Dataset and Split Mapping

- Dataset root: `/home/adhil/github/EMULaToR/data/processed/baselines/MPEK_assayed`.
- Task: `kcat`.
- Split axes:
  - `enzyme_sequence_splits/threshold_0.01`
  - `uniprot_time_splits`
- Condition-aware and condition-disabled runs use identical expanded rows.
- Full experiments use seeds 666, 777, and 888.
- The existing collapsed-data MPEK results are a contextual reference, not a controlled input ablation.

## 6. Files Under `assay_emulator_bench/`

- `dataset.py`: validate and load assay fields, convert units, impute range artifacts, and retain pair/provenance metadata.
- `modeling.py`: load and call MPEK's native auxiliary encoder without importing the Mole-BERT package.
- `train_single_split.py`: pass conditions to the model, calculate row and pair metrics, select checkpoints, and save prediction artifacts.
- Run, launch, tuning, and aggregation entrypoints: propagate assay and pair-reduction options.
- `tests/`: cover condition preprocessing, native condition encoding, pair aggregation, checkpoint selection, and backward compatibility.
- `README.md` and `commands.txt`: document the paired ablation and exact commands.

## 7. Minimal Edits Outside `assay_emulator_bench/`

- Update this `Plan.md` only.
- Do not modify `MTLKcatKM`, Mole-BERT, datasets, embedding caches, or existing result directories.

## 8. Exact Execution Plan

1. Use Conda environment `mldb`.
2. Validate the expanded P-seq and Time schemas and pair membership.
3. Reuse the complete existing protein and ligand caches.
4. Implement condition-aware and condition-disabled paths behind backward-compatible CLI flags.
5. Save assay-row and maximum-per-pair metrics and predictions in separate artifacts.
6. Run targeted unit tests and import checks.
7. Run tiny condition-aware and condition-disabled P-seq smoke tests with `CUDA_VISIBLE_DEVICES=0`, the default 50 epochs, and reduced row counts only.
8. Validate output schemas, condition diagnostics, pair denominators, source-file hashes, and `git diff --check`.
9. Provide commands for the 12 full paired-ablation runs without launching them.
