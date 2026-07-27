# MPEK assay_emulator_bench

This bench retrains MPEK on the assay-expanded EMULaToR kcat data. It supports MPEK's native pH/temperature encoder and a condition-disabled control on the same rows.

Default data root:

```text
/home/adhil/github/EMULaToR/data/processed/baselines/MPEK_assayed
```

The default task is `kcat`. The default axes are P-seq at `threshold_0.01` and Time:

```text
kcat/enzyme_sequence_splits/threshold_0.01
kcat/uniprot_time_splits
```

## Inputs

Each split file is expected to contain:

- `sequence`: enzyme amino-acid sequence
- `smiles`: substrate SMILES
- `log10_value`: regression target

The P-seq and Time files include:

- `pH`
- `Temperature` in kelvin
- `assay_mapping_status`
- source-imputation flags
- one or more assay rows per `(sequence, smiles)` pair

Use `--use_assay_conditions` to feed pH and temperature to MPEK. The wrapper converts temperature to Celsius and uses MPEK's native RBF condition encoder. Organism input remains disabled.

The wrapper replaces clear condition-extraction artifacts before encoding:

- pH outside `[0,14)` becomes `7.0`.
- Temperature below 0 or above 100 degrees Celsius becomes `25.0`.
- Exactly 100 degrees Celsius remains unchanged.

`condition_summary.json` records source imputations, range-based replacements, and raw/final ranges for each partition.

## Embeddings

Protein embeddings:

- Encoder: ProtT5 from `Rostlab/prot_t5_xl_uniref50`
- Cache value: original 1024-dimensional mean-pooled embedding
- Cache path: `embeddings/proteins/<hash-prefix>/<hash>.npz`

Substrate embeddings:

- Encoder: upstream `Mole-BERT` submodule
- Checkpoint: `Mole-BERT/model_gin/Mole-BERT.pth`
- Cache value: 300-dimensional mean-pooled graph embedding
- Cache path: `embeddings/ligands/molebert/<hash-prefix>/<hash>.npz`

`prepare_assets.py --auto` validates the checkpoint and can recover it from the submodule history if it is missing.

## Model

The cached-input model follows the local MPEK implementation:

- ProtT5 1024-d protein vector
- learned `Linear(1024, 300)` projection
- Mole-BERT 300-d ligand vector
- no-assay input: ligand 300 + projected protein 300 = `600`
- assay-aware input: ligand 300 + projected protein 300 + native condition 300 = `900`
- original `MTLKcatKM.layers.PLE` and `MLP` blocks

MPEK's native condition encoder applies separate RBF banks to pH and temperature, projects each bank to 300 dimensions, and sums the projections. The wrapper loads the upstream `auxiliary_encoder.py` directly. It does not modify or duplicate the encoder.

Default training settings follow the paper-style configuration:

- `lr=1e-4`
- `num_experts=4`
- `expert_layers=1`
- `expert_dim=768`
- `dropout=0.2`
- `batch_size=1`
- `epochs=50`
- `tower_layers=3`
- `tower_hidden=128`

CUDA runs enable TF32 and automatic mixed precision. AMP uses bf16 on Ampere-or-newer GPUs and fp16 otherwise.

## Commands

The commands use these defaults:

```text
data:       /home/adhil/github/EMULaToR/data/processed/baselines/MPEK_assayed
embeddings: /home/adhil/github/EMULaToR/data/processed/baselines/MPEK/embeddings
task:       kcat
axes:       enzyme_sequence_splits and uniprot_time_splits
seeds:      666, 777, and 888
```

The shared cache already covers the expanded pairs. This command verifies coverage and writes only missing features:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mldb \
  python assay_emulator_bench/cache_embeddings.py \
  --device cuda:0
```

## Assay-aware paired ablation

The condition-aware and condition-disabled runs use identical expanded rows. Both commands use `--pair_reducer max`, which:

- trains with assay-row MSE;
- records assay-row and pair-level validation metrics;
- selects the best checkpoint with pair-level validation RMSE;
- takes the maximum true and predicted log10 value for each `(sequence, smiles)` pair;
- writes pair-level metrics to `final_results_<split>.csv`;
- writes assay-row metrics to `final_results_<split>_assay_rows.csv`;
- writes both prediction levels as Parquet files.

Run condition-aware MPEK:

```bash
conda run --no-capture-output -n mldb \
  python assay_emulator_bench/launch_parallel_retrain.py \
  --gpus 0 1 2 3 4 5 \
  --runs_per_gpu 1 \
  --results_dirname mpek_results_assay_aware \
  --skip_cache \
  --use_assay_conditions \
  --temperature_unit kelvin \
  --pair_reducer max \
  --default_settings
```

Run the matched condition-disabled control:

```bash
conda run --no-capture-output -n mldb \
  python assay_emulator_bench/launch_parallel_retrain.py \
  --gpus 0 1 2 3 4 5 \
  --runs_per_gpu 1 \
  --results_dirname mpek_results_assay_blind_expanded \
  --skip_cache \
  --temperature_unit kelvin \
  --pair_reducer max \
  --default_settings
```

Run condition-aware MPEK sequentially on physical GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n mldb \
  python assay_emulator_bench/run_split_benchmarks.py \
  --results_dirname mpek_results_assay_aware \
  --skip_cache \
  --use_assay_conditions \
  --temperature_unit kelvin \
  --pair_reducer max \
  --device cuda:0
```

Change `--gpus` to the available physical GPU IDs. The two parallel commands create 12 full runs in total and do not overwrite the submitted MPEK results.

Aggregate condition-aware results:

```bash
conda run --no-capture-output -n mldb \
  python assay_emulator_bench/aggregate_mpek_results.py \
  --results_dirname mpek_results_assay_aware
```

Aggregate condition-disabled results:

```bash
conda run --no-capture-output -n mldb \
  python assay_emulator_bench/aggregate_mpek_results.py \
  --results_dirname mpek_results_assay_blind_expanded
```

## Resumability

- Embedding cache files are skipped when present.
- Cache writes use temporary files followed by atomic rename.
- Training writes `checkpoint_latest.pt` every epoch.
- Restarted runs resume from `checkpoint_latest.pt`.
- Resume fails clearly if assay mode or pair reduction differs from the saved checkpoint.
- Completed runs with `final_results_test.csv` are skipped unless `--overwrite` is passed.
- Optuna studies use SQLite and `load_if_exists=True`.
