# Adaptive SAND

This repository provides a modular and reproducible implementation of Adaptive SAND and Hybrid Adaptive SAND experiments for feature selection and classification. The codebase is organized so that a user can clone the repository, install the dependencies, place the datasets in the expected directory, and run a selected model or a full comparison from the command line.

The project was refactored from an experimental script into a GitHub-ready structure. Model definitions, pruning callbacks, dataset loading utilities, experiment runners, logging, and plotting are separated into dedicated modules.

## Repository Structure

```text
adaptive-sand/
├── main.py
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── adaptive_sand/
│   ├── __init__.py
│   ├── callbacks/
│   │   ├── __init__.py
│   │   └── pruning.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── transforms.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── compare.py
│   │   └── train.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── adaptive.py
│   │   ├── baselines.py
│   │   └── hybrid.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── plotting.py
│       └── reproducibility.py
├── datasets/
│   └── mice
|   └── coil
|   └── isolet
|   └── har70
|
└── sand/
    └── experiments...
```

## Main Components

- **`main.py`**: Command-line entry point for running experiments.
- **`adaptive_sand/models/adaptive.py`**: First-order Adaptive SAND model.
- **`adaptive_sand/models/hybrid.py`**: Hybrid Adaptive SAND model using first-order and second-order pruning signals.
- **`adaptive_sand/callbacks/pruning.py`**: Smart pruning callbacks for adaptive and hybrid variants.
- **`adaptive_sand/experiments/compare.py`**: Experiment runner for single-model runs and multi-model comparisons.
- **`adaptive_sand/data/loader.py`**: Dataset loader wrapper that connects the repository-local `datasets/` directory to the original SAND dataset interface.
- **`adaptive_sand/utils/`**: Utilities for reproducibility, CSV logging, and plotting.
- **`configs/default.yaml`**: Default experiment configuration.

## Supported Models

The following model options are available through the `--model` command-line argument:

```text
original   Original SAND baseline from the external SAND package
seqatt     Sequential Attention baseline from the external SAND package
adaptive   Adaptive SAND with first-order gradient-based pruning
hybrid     Hybrid Adaptive SAND with first-order and second-order pruning signals
lasso      LASSO feature selection followed by an MLP classifier
all        Run all available models and compare them
```

## Requirements

The project is tested with Python 3.10. A pinned dependency stack is provided in `requirements.txt` to avoid NumPy/TensorFlow/SciPy binary incompatibility issues.

```text
numpy==1.23.5
pandas==1.5.3
scipy==1.10.1
scikit-learn==1.2.2
matplotlib==3.7.3
tensorflow==2.10.1
PyYAML==6.0.1
```

Important: TensorFlow 2.10.x, SciPy, and Pandas wheels may fail with NumPy 2.x. If you see an error such as `numpy.dtype size changed` or `_ARRAY_API not found`, create a clean virtual environment and reinstall the pinned requirements.

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd adaptive-sand
```

Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If you previously installed incompatible versions globally, do not reuse that environment. Use a clean virtual environment instead. If needed, the current environment can be reset with:

```bash
pip uninstall -y numpy pandas scipy scikit-learn tensorflow keras matplotlib
pip install -r requirements.txt
```

## External SAND Dependency

This project keeps compatibility with the original SAND implementation and expects the following import paths to be available:

```python
from sand.experiments.datasets.dataset import get_dataset
from sand.experiments.models.mlp_sand import SANDModel
from sand.experiments.models.mlp_sa import SequentialAttentionModel
```

Therefore, the original `sand/` directory should be available at the repository root:

```text
adaptive-sand/
├── adaptive_sand/
├── datasets/
├── sand/
│   └── experiments/
│       ├── datasets/
│       └── models/
└── main.py
```

If SAND is maintained as a separate repository, it can also be added as a Git submodule:

```bash
git submodule add <sand-repo-url> sand
git submodule update --init --recursive
```

## Dataset Placement

Datasets should be placed under the repository-local `datasets/` directory:

```text
adaptive-sand/
├── datasets/
│   ├── mice/
│   │   └── Data_Cortex_Nuclear.csv
│   ├── isolet/
│   ├── coil/
│   ├── activity/
│   └── mnist/
└── main.py
```

By default, the project automatically sets:

```text
SAND_DATA_DIR=<repo-root>/datasets
```

Some legacy SAND dataset loaders build file paths directly under `sand/experiments/datasets/<dataset>` and do not read `SAND_DATA_DIR`. To avoid modifying the upstream SAND code, `adaptive_sand/data/loader.py` automatically creates a directory alias from:

```text
datasets/<dataset>
```

to:

```text
sand/experiments/datasets/<dataset>
```

On systems where symlink creation is not allowed, such as some Windows configurations, the dataset directory is copied automatically to the legacy SAND path.

You can also specify a custom dataset root manually:

```bash
python main.py --dataset mice --model hybrid --data-dir /path/to/datasets
```

On Windows PowerShell:

```powershell
python main.py --dataset mice --model hybrid --data-dir F:\path\to\datasets
```

## Dataset Loader Contract

The original SAND dataset interface is expected to return a dictionary with the following keys:

```python
{
    "ds_train": ds_train,
    "ds_val": ds_val,
    "ds_test": ds_test,
    "num_classes": num_classes,
    "num_features": num_features,
    "is_classification": True,
}
```

To add a new dataset, extend the original SAND dataset registry, usually located at:

```text
sand/experiments/datasets/dataset.py
```

Example:

```python
if name == "my_dataset":
    return load_my_dataset(val_ratio=val_ratio, batch_size=batch_size)
```

Then run:

```bash
python main.py --dataset my_dataset --model hybrid
```

## Quick Start

Run Hybrid Adaptive SAND on the MICE dataset:

```bash
python main.py --dataset mice --model hybrid --epochs 100
```

Run a short smoke test:

```bash
python main.py --dataset mice --model hybrid --epochs 5 --skip-plots
```

Run Adaptive SAND:

```bash
python main.py --dataset mice --model adaptive --epochs 100
```

Run all available models:

```bash
python main.py --dataset mice --model all --epochs 100
```

Use the default YAML configuration:

```bash
python main.py --config configs/default.yaml
```

Override selected settings from the command line:

```bash
python main.py --config configs/default.yaml --dataset mice --model hybrid --epochs 50 --skip-plots
```

## Command-Line Arguments

```text
--config       Optional YAML configuration file
--dataset      Dataset name, e.g. mice, mnist, isolet, coil, activity
--model        Model to run: original, seqatt, adaptive, hybrid, lasso, all
--epochs       Number of training epochs
--batch-size   Batch size
--val-ratio    Validation split ratio passed to the SAND dataset loader
--data-dir     Optional custom dataset root directory
--initial-k    Initial number of active features for adaptive/hybrid models
--output-dir   Directory for CSV results and plots
--seed         Random seed
--skip-plots   Disable plot generation
```

## Configuration File

The default configuration is stored in:

```text
configs/default.yaml
```

Example:

```yaml
dataset: mnist
model: hybrid
epochs: 150
batch_size: 32
val_ratio: 0.125
data_dir: null
seed: 42
initial_k: null
output_dir: outputs
skip_plots: false

model_params:
  hidden_layers: [67]
  learning_rate: 0.0001
  decay_steps: 250
  decay_rate: 1.0
  alpha: 0.0
  batch_norm: false
  original_sand_k: 60
  seqatt_k: 60
  lasso_k: 60

pruning:
  start_epoch: 5
  interval: 4
  tolerance: 0.005
  lambda_k: 0.15
  hybrid_alpha: 0.7
  min_val_accuracy: 0.95
  cooldown: 3
  max_rejections: 3
```

Dataset-specific configurations can be added under `configs/`, for example:

```text
configs/mice.yaml
configs/mnist.yaml
configs/isolet.yaml
```

Then run:

```bash
python main.py --config configs/mice.yaml
```

## Pruning Parameters

Adaptive and Hybrid Adaptive SAND use smart pruning callbacks. The main pruning parameters are:

- **`start_epoch`**: First epoch at which pruning is allowed.
- **`interval`**: Number of epochs between pruning attempts.
- **`tolerance`**: Allowed score tolerance when deciding whether to accept or reject pruning.
- **`lambda_k`**: Penalty weight for retaining many features.
- **`hybrid_alpha`**: Weight assigned to first-order importance in the hybrid pruning score. The second-order component receives `1 - hybrid_alpha`.
- **`min_val_accuracy`**: Minimum validation accuracy required before pruning can be attempted.
- **`cooldown`**: Minimum number of epochs between pruning attempts.
- **`max_rejections`**: Maximum number of rejected pruning attempts from the same feature count.

For small datasets such as MICE, pruning can be sensitive to validation accuracy fluctuations. If pruning appears too aggressive, consider using a dataset-specific configuration with a higher `min_val_accuracy`, a later `start_epoch`, or a smaller `lambda_k`.

Example:

```yaml
pruning:
  start_epoch: 30
  interval: 5
  tolerance: 0.005
  lambda_k: 0.10
  hybrid_alpha: 0.7
  min_val_accuracy: 0.99
  cooldown: 5
  max_rejections: 3
```

## Outputs

By default, outputs are saved under:

```text
outputs/
```

The main output files are:

```text
outputs/results_summary.csv
outputs/training_summary.png
```

The CSV file contains one row per model run and includes:

```text
dataset
model
accuracy
loss
time_sec
ram_mb
selected_k
```

If `--skip-plots` is provided, plot generation is disabled.

## Reproducibility

The project sets seeds for Python, NumPy, and TensorFlow:

```bash
python main.py --dataset mice --model hybrid --seed 42
```

For academic reporting, it is recommended to run multiple seeds and report mean and standard deviation:

```bash
python main.py --dataset mice --model hybrid --seed 1
python main.py --dataset mice --model hybrid --seed 2
python main.py --dataset mice --model hybrid --seed 3
python main.py --dataset mice --model hybrid --seed 4
python main.py --dataset mice --model hybrid --seed 5
```

Note that exact reproducibility may still depend on hardware, TensorFlow backend behavior, and GPU/CPU execution differences.

## Fair Comparison Notes

The experiment runner uses shared settings such as epochs, batch size, learning rate, hidden layer size, and dataset splits across models where applicable. However, for formal academic reporting, the following points should be considered:

- **Multiple seeds**: Report mean and standard deviation over multiple random seeds.
- **Fixed vs adaptive feature budgets**: Original SAND, Sequential Attention, and LASSO may use a fixed `k`, while Adaptive and Hybrid SAND determine `k` dynamically. This should be explicitly described in any paper or report.
- **Normalization**: The current implementation applies batch-wise normalization for compatibility with the original script. For stricter experimental control, train-set normalization statistics can be computed once and reused for validation and test sets.
- **Runtime and memory**: Timing and memory results can be affected by TensorFlow warm-up, graph tracing, and hardware differences.
- **Small datasets**: On small datasets, pruning decisions may be sensitive to validation split variance. Dataset-specific pruning configurations are recommended.

## CPU and GPU Notes

If TensorFlow prints messages such as:

```text
Could not load dynamic library 'cudart64_110.dll'
Could not load dynamic library 'nvcuda.dll'
```

this usually means that TensorFlow did not find a CUDA-enabled GPU setup and will run on CPU. These warnings are not fatal if CPU execution is acceptable.

## Common Issues

### NumPy binary incompatibility

Error examples:

```text
AttributeError: _ARRAY_API not found
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

Solution:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset file not found under `sand/experiments/datasets`

If a legacy SAND loader searches under:

```text
sand/experiments/datasets/<dataset>
```

make sure your actual data exists under:

```text
datasets/<dataset>
```

The wrapper should automatically create an alias or copy the dataset to the legacy path. If this fails, check write permissions in the repository directory.

### Import error for `sand`

If you see:

```text
Could not import sand.experiments.datasets.dataset.get_dataset
```

make sure the original SAND implementation is available as:

```text
adaptive-sand/sand/
```

or installed as an importable Python package.

## Suggested Workflow for a New User

1. Clone the repository.
2. Create a clean Python 3.10 virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Place the original SAND code under `sand/`.
5. Place datasets under `datasets/`.
6. Run a short smoke test:

```bash
python main.py --dataset mice --model hybrid --epochs 5 --skip-plots
```

7. Run the full experiment:

```bash
python main.py --dataset mice --model hybrid --epochs 100
```

8. Check the output CSV:

```text
outputs/results_summary.csv
```

## License and Dataset Notice

Before publishing this repository, verify the license of the original SAND code and the datasets. If dataset redistribution is not allowed, keep only `datasets/README.md` in the repository and provide download instructions instead of committing raw dataset files.

## Citation

If this repository is used in academic work, cite the original SAND method and any dataset sources according to their respective licenses and citation guidelines.

