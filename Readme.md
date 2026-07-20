# Adaptive-SAND

Implementation and experimental evaluation of **Adaptive SAND (A-SAND)** and
**A-SAND-Hybrid** for neural feature selection.

This repository provides a notebook-based experimental framework that compares
the proposed methods with **Original SAND** and an **L1-Logistic Regression +
MLP** baseline. The goal is to retain classification performance while
automatically selecting a substantially smaller subset of input features.

## Project Overview

High-dimensional datasets may contain irrelevant or redundant features.
Training a neural network using every feature can increase computational cost,
reduce interpretability, and affect generalisation.

Adaptive-SAND integrates feature selection into neural-network training:

- **A-SAND** estimates feature importance from first-order gradient information.
- **A-SAND-Hybrid** combines first-order gradients with second-order curvature
  information estimated through a Hutchinson-style diagonal Hessian estimator.
- **Original SAND** is evaluated as a reference baseline.
- **L1-Logistic Regression + MLP** is included as a conventional
  feature-selection baseline.

The complete workflow is implemented in `main.ipynb`.

## Methods

### A-SAND

A-SAND applies trainable weights to the input features before the classifier.
It accumulates gradient-based importance scores during training and prunes
features with low importance after validation accuracy reaches a plateau.

Pruned feature weights are masked permanently, so removed features cannot
re-enter the model later in training.

### A-SAND-Hybrid

A-SAND-Hybrid augments gradient importance with a second-order feature score.
The diagonal Hessian is estimated stochastically using a Hutchinson-style
estimator, then combined with the first-order score:

\[
S_{\text{hybrid}} =
\alpha S_{\text{first-order}} +
(1 - \alpha) S_{\text{second-order}}
\]

- `alpha = 1.0`: only first-order gradient information
- `alpha = 0.0`: only second-order Hessian information
- `alpha = 0.7`: hybrid setting used in the default notebook configuration

A-SAND-Hybrid is generally slower than A-SAND because it computes an additional
second-order approximation during training.

### Plateau-Aware Pruning

Both proposed methods use a validation-plateau-based pruning strategy.

1. Training starts with all input features active.
2. Feature importance is accumulated during training.
3. Once validation performance stabilises, low-scoring features are pruned.
4. The pruned model is evaluated in the next epoch.
5. Pruning is retained if validation performance remains within the accepted
   tolerance; otherwise, the previous model weights, mask, and feature count
   are restored.
6. The best accepted state is restored at the end of training.

## Repository Structure

```text
Adaptive-Sand/
├── datasets/            # Dataset folder placeholder; raw datasets are excluded
├── notebooks/           # Additional notebook files, if any
├── results/             # CSV results, figures, and sensitivity-analysis outputs
├── sand/                # Original SAND baseline and dataset-loading utilities
├── adaptive_sand.py     # First-order A-SAND implementation
├── main.ipynb           # Main experiment and analysis notebook
├── requirements.txt     # Python dependencies
├── .gitignore
└── Readme.md
```

## Notebook Workflow

`main.ipynb` is the central executable component of the project. It includes:

- Google Drive mounting and project-path configuration
- Deterministic seed configuration
- Dataset loading
- A-SAND training and adaptive pruning
- A-SAND-Hybrid training and adaptive pruning
- Original SAND baseline experiments
- L1-Logistic Regression + MLP baseline experiments
- Test accuracy, feature count, runtime, CPU RAM, and GPU-memory measurement
- CSV export of comparison results
- Accuracy and feature-retention plots
- Hyperparameter sensitivity and robustness analysis

The comparison notebook evaluates Original SAND and L1-based baselines at fixed
and feature-count-matched settings derived from the proposed methods. [file:25]

## Installation

Clone the repository:

```bash
git clone https://github.com/eren9763/Adaptive-Sand.git
cd Adaptive-Sand
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The project is designed to be executed from the notebook:

```text
main.ipynb
```

### Google Colab

1. Upload or clone this repository into Google Drive.
2. Upload the required datasets separately, as explained below.
3. Open `main.ipynb` in Google Colab.
4. Enable a GPU from **Runtime -> Change runtime type -> T4 GPU** or another
   available GPU.
5. Update `ROOT_PATH` in the first notebook cell to match your Google Drive
   location.
6. Run the notebook cells from top to bottom.

The default notebook uses a Google Drive root similar to:

```python
ROOT_PATH = "/content/drive/MyDrive/SAND"
```

If your folder name or location differs, update this variable before running the
remaining cells.

### Local Jupyter

After installing the dependencies, start Jupyter:

```bash
jupyter notebook
```

Then open `main.ipynb`, update the path-related notebook cells if necessary,
and run all cells in order.

## Dataset Availability

> **Raw datasets are not included in this repository.**

The dataset collection used in this project is approximately **500 MB**, so it
could not be uploaded to GitHub. Some datasets may also have separate licenses
or redistribution conditions.

The repository retains the `datasets/` directory only to document the intended
project layout. To reproduce experiments, download the datasets independently
from their original public sources and place them in your own local or Google
Drive dataset directory.

The notebook supports dataset identifiers including:

```text
mnist
mice
isolet
coil
arcene
har70
```

For Google Drive usage, the intended layout is:

```text
MyDrive/
└── SAND/
    ├── adaptive_sand.py
    ├── main.ipynb
    ├── sand/
    ├── results/
    └── datasets/
        ├── har70/
        ├── coil/
        ├── isolet/
        ├── mice/
        └── arcene/
```

The data loader used by the notebook is located in:

```text
sand/experiments/datasets/
```

Ensure that each dataset has the exact file and folder layout expected by that
loader.

### Google Drive Note

The notebook includes a symbolic-link approach intended to connect the dataset
folder with the location expected by the loader. Google Drive may not support
symbolic links in Colab, which can cause an `Operation not supported` error.

If this occurs, use one of these alternatives:

- Copy the datasets into the exact directory expected by
  `sand/experiments/datasets/`.
- Modify the dataset loader path to point directly to your dataset directory.
- Copy the required data temporarily into the Colab runtime storage.

## Configuration

The main experiment configuration is located in `main.ipynb`. Before running,
review and update the following values as needed:

```python
DATASET_NAME = "mnist"
EPOCHS = 150
FIXED_K = 60
LAYER_SEQUENCE = (67,)
HYBRID_ALPHA = 0.7
SEED = 42
```

The notebook also includes a hyperparameter sensitivity analysis for:

- Pruning rate
- Sparsity penalty
- Cooldown period
- Plateau tolerance
- Random seed

## Results

Generated outputs are saved under `results/`.

The notebook records:

- Test accuracy
- Final selected-feature count, `k`
- Feature-retention ratio, `k / d`
- Training time
- CPU resident-memory usage
- GPU peak-memory usage, when available
- Number of accepted or attempted pruning steps

Typical generated files include:

```text
results/unified_comparison_<dataset>.csv
results/<dataset>_comparison.png
results/<dataset>_sensitivity.csv
results/<dataset>_sensitivity_summary.csv
results/<dataset>_robustness.csv
results/<dataset>_sensitivity_comparison.png
results/<dataset>_sensitivity_detail.png
```

## Reproducibility

The notebook fixes random seeds for Python, NumPy, and TensorFlow. It also
enables TensorFlow deterministic operations where supported. For comparable
results, keep the dataset split, seed, architecture, epoch count, and pruning
parameters unchanged. [file:25]

Exact numerical equality may still vary across operating systems, TensorFlow
versions, CUDA versions, and GPU models.

## Important Notes

- `main.ipynb` is the main entry point; there is no separate command-line
  experiment script in this version of the repository.
- `adaptive_sand.py` contains the first-order A-SAND implementation.
- The A-SAND-Hybrid implementation and its pruning callbacks are defined in
  `main.ipynb`.
- `sand/` must be retained because the notebook imports its Original SAND model
  and dataset loader.
- `datasets/` is retained as a project folder, but the raw dataset files are
  not distributed through this repository.
- `results/` contains experiment outputs and is intentionally retained.
- A GPU is recommended, especially for A-SAND-Hybrid and sensitivity sweeps.

## License

A license has not yet been specified. Add a `LICENSE` file before reusing or
redistributing this code.
