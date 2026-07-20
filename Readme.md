# Adaptive-SAND

TensorFlow/Keras implementation of **Adaptive SAND (A-SAND)** and
**A-SAND-Hybrid** for adaptive neural feature selection.

This repository contains the implementation, experiment configuration, dataset
loading utilities, notebooks, and selected experimental results used to compare
A-SAND variants with Original SAND and conventional feature-selection baselines.

---

## Overview

High-dimensional datasets often contain redundant or irrelevant input features.
Training a neural network with all features can increase computational cost,
reduce interpretability, and harm generalisation.

This project proposes adaptive feature-selection mechanisms that are integrated
directly into neural-network training:

- **A-SAND** performs adaptive feature selection using first-order
  gradient-based importance estimates.
- **A-SAND-Hybrid** extends A-SAND by combining first-order information with a
  stochastic, second-order diagonal Hessian approximation.
- **Original SAND** is retained as a baseline method.
- A baseline MLP and L1-Logistic Regression + MLP experiments can also be used
  for comparative evaluation.

The main objective is to preserve predictive performance while reducing the
number of selected input features.

---

## Methods

### A-SAND

A-SAND uses a learnable feature-weight vector and a binary feature mask before
the classifier layers.

During training:

1. Each input feature is scaled by its learnable selection weight.
2. Gradient-based importance scores are accumulated for active features.
3. When validation accuracy plateaus, low-importance features are pruned.
4. The feature mask permanently removes pruned weights from the model.
5. The pruning step is accepted or rolled back depending on validation
   performance.

### A-SAND-Hybrid

A-SAND-Hybrid incorporates both first- and second-order feature importance.

- **First-order score:** Gradient-based importance.
- **Second-order score:** A diagonal Hessian estimate obtained with a
  Hutchinson-style stochastic estimator.
- **Hybrid score:** A weighted combination of normalised first- and
  second-order scores.

\[
S_{\text{hybrid}} =
\alpha S_{\text{first-order}} +
(1 - \alpha) S_{\text{second-order}}
\]

where:

- `alpha = 1.0` uses only first-order importance.
- `alpha = 0.0` uses only second-order importance.
- `alpha = 0.7` is the default hybrid setting used in the experiments.

Because it estimates second-order information during training, A-SAND-Hybrid is
expected to require more computation time than first-order A-SAND.

### Plateau-aware Pruning

Both methods use plateau-aware pruning callbacks.

A pruning decision is triggered only after validation accuracy has stabilised.
The model state before pruning is saved, and the pruning result is evaluated in
the following epoch:

- The new mask is retained if validation performance remains within the allowed
  tolerance.
- The model weights, feature mask, and selected-feature count are restored if
  performance degrades excessively.
- The best accepted model state is restored at the end of training.

---

## Repository Structure

```text
Adaptive-Sand/
├── adaptive_sand/          # Proposed A-SAND and A-SAND-Hybrid modules
│   ├── __init__.py
│   ├── layers.py           # Feature-selection layer implementations
│   ├── models.py           # A-SAND and A-SAND-Hybrid models
│   ├── callbacks.py        # Plateau-aware pruning callbacks
│   └── utils.py            # Reproducibility and utility functions
│
├── sand/                   # Current Original SAND code and dataset utilities
├── configs/                # Experiment configurations
├── datasets/               # Dataset directory structure and loader resources
├── notebooks/              # Google Colab and exploratory notebooks
├── results/                # Result CSV files, figures, and experiment outputs
├── run_experiment.py       # Main command-line experiment entry point
├── requirements.txt        # Required Python packages
└── README.md
```

### Important folders

| Folder | Purpose |
|---|---|
| `adaptive_sand/` | Proposed A-SAND and A-SAND-Hybrid implementations |
| `sand/` | Current Original SAND baseline and dataset-loading infrastructure |
| `datasets/` | Expected local dataset location; raw data are not distributed here |
| `results/` | Experiment CSV files, figures, and reported outputs |
| `notebooks/` | Google Colab workflow and exploratory analyses |
| `configs/` | Dataset and experiment settings |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/eren9763/Adaptive-Sand.git
cd Adaptive-Sand
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- NumPy
- Pandas
- scikit-learn
- psutil
- Matplotlib

A CUDA-compatible GPU is recommended, especially for A-SAND-Hybrid, which
computes a second-order approximation during training.

---

## Dataset Availability

> **Important:** The raw datasets are not included in this GitHub repository.

The dataset collection is approximately **500 MB**, which makes it unsuitable
for normal GitHub storage and distribution. In addition, some datasets may have
their own terms of use or redistribution restrictions.

To reproduce the experiments, download the datasets from their original public
sources and place them in the expected local dataset structure.

The experimental workflow uses dataset identifiers such as:

```text
mnist
mice
isolet
coil
arcene
har70
```

The dataset-loading code is located under:

```text
sand/experiments/datasets/
```

The expected dataset paths should be configured according to the loader logic in
that directory.

### Google Colab / Google Drive

For Google Colab experiments, a practical layout is:

```text
MyDrive/SAND/
├── Adaptive-Sand/          # This repository
└── datasets/               # Downloaded raw datasets
```

Mount Google Drive and add the repository root to Python's import path:

```python
from google.colab import drive
import sys

drive.mount("/content/drive")

ROOT_PATH = "/content/drive/MyDrive/SAND/Adaptive-Sand"
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
```

> Google Drive may not support symbolic links created with `ln -s` in Colab.
> If symbolic linking fails, copy the required dataset to the expected path or
> update the dataset root path in the loader configuration.

---

## Running Experiments

### Run A-SAND and A-SAND-Hybrid

```bash
python run_experiment.py --dataset mnist --epochs 150 --model both --seed 42
```

### Run only A-SAND

```bash
python run_experiment.py --dataset mnist --epochs 150 --model asand --seed 42
```

### Run only A-SAND-Hybrid

```bash
python run_experiment.py --dataset mnist --epochs 150 --model hybrid --seed 42
```

### Run the baseline MLP

```bash
python run_experiment.py --dataset mnist --epochs 150 --model baseline --seed 42
```

### Run all enabled models

```bash
python run_experiment.py --dataset mnist --epochs 150 --model all --seed 42
```

Display all supported arguments:

```bash
python run_experiment.py --help
```

---

## Example Imports

The proposed implementations can be imported directly from the
`adaptive_sand` package:

```python
from adaptive_sand import (
    SANDAdaptiveModel,
    SANDAdaptiveModelHybrid,
    SANDSmartPruningCallback,
    SANDSmartHybridPruningCallback,
    set_seeds,
)
```

For first-order A-SAND:

```python
model = SANDAdaptiveModel(
    num_inputs=num_features,
    num_outputs=num_classes,
    initial_k=num_features,
    layer_sequence=(67,),
    is_classification=True,
)

model.compile(metrics=["accuracy"])
```

For A-SAND-Hybrid:

```python
model = SANDAdaptiveModelHybrid(
    num_inputs=num_features,
    num_outputs=num_classes,
    initial_k=num_features,
    layer_sequence=(67,),
    is_classification=True,
)

model.compile()
```

---

## Outputs

Experiment outputs are written under `results/`.

Typical recorded metrics include:

- Test accuracy
- Final selected-feature count, `k`
- Feature-retention ratio, `k / d`
- Number of pruning steps
- Training time in seconds
- CPU resident memory usage
- GPU peak-memory usage, when available

The experiment notebook compares A-SAND and A-SAND-Hybrid with Original SAND
at matched feature counts, as well as an L1-Logistic Regression + MLP baseline. [file:2]

---

## Reproducibility

The project includes deterministic settings for Python, NumPy, and TensorFlow.
It sets `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`, and
`TF_CUDNN_DETERMINISTIC`, then resets random seeds before each model run. [file:2]

For a fair comparison, keep the following fixed across runs:

- Dataset split
- Random seed
- Batch size
- Number of epochs
- Model architecture
- Pruning parameters
- TensorFlow and CUDA versions

Exact numerical equality is not guaranteed across different operating systems,
TensorFlow versions, CUDA versions, or GPU models.

---

## Notes

- `adaptive_sand/` contains the proposed methods; `sand/` contains the current
  Original SAND code and data-loading utilities.
- Do not remove `sand/` unless you also replace the imports used for the
  Original SAND baseline and dataset loader.
- A-SAND-Hybrid is generally slower than A-SAND because it computes an
  additional Hessian-based importance estimate.
- Raw datasets are intentionally excluded from GitHub; see
  [Dataset Availability](#dataset-availability).
- The `results/` directory is retained to provide experimental outputs and
  figures.

---

## Citation

If you use this code, please cite the accompanying manuscript when it becomes
available.

```bibtex
@software{koybasi2026adaptivesand,
  author  = {Eren Köybaşı},
  title   = {Adaptive-SAND: Adaptive Neural Feature Selection},
  year    = {2026},
  url     = {https://github.com/eren9763/Adaptive-Sand}
}
```

---

## License

A license has not yet been specified. Add a `LICENSE` file before reusing or
redistributing the code.
