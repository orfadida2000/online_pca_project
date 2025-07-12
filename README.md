# Online PCA Project

This repository implements online Principal Component Analysis (PCA) algorithms, including Sanger's rule and
simultaneous Oja's rule, for learning principal components from streaming data. It provides modular training,
validation, and visualization tools for analyzing learned components and their generalization performance.

---

## Features

- Online learning of PCA components using Sanger's and Oja's rules
- Modular training and validation pipeline
- Visualization of samples, learned components, generalization errors, and angle trajectories
- Configurable learning rate schemas and stopping rules
- Support for batch and debug modes

---

## 📁 Project Structure

```
pca_project/
│
├── online_pca/
│   ├── __init__.py
│   ├── utilities.py                   # Sample generation, stopping rule, etc
│   ├── visualization.py               # Plotting tools
│   ├── learning/
│   │   ├── __init__.py
│   │   ├── learning_rate_schemas.py   # Defines learning rate functions
│   │   └── training.py                # Core `run_online_pca` function
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── sanger_rule.py             # Sanger-specific PCA rule
│   │   └── simultaneous_oja_rule.py   # Simultaneous Oja rule  
│   └── validation/
│       ├── __init__.py
│       ├── initialization.py          # Initialization logic for plotting
│       └── update.py                  # Angle/gen. error update utilities
│
├── pca_assignment/
│   ├── .ipynb_checkpoints/
│   ├── assignment_code.py             # Main script to run experiments
│   └── interactive_plot.ipynb         # Jupyter for interactive visualizations
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt                   # Python dependencies
```

---

## 🚀 Running the Project

### ✔ Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

### ▶ How to Run

**Always run `assignment_code.py` as the main script**  
Make sure the **current working directory is the root folder (`PCA Project`)**.

```bash
python PCA\ Assignment/assignment_code.py
```

### ⚙ Modes

Enable modes by passing CLI arguments:

- `--debug` — Extra printing: all eigenvectors, found components, average sample, best achievable generalization error.
- `--batch` — Skips `plt.show()`; saves plots silently.

Example:
```bash
python PCA\ Assignment/assignment_code.py --debug --batch
```

---

## 📌 Key Components

### ✴ `run_online_pca(...)` (in `training.py`)
Generic online PCA loop. Takes:
- `delta_fn`: Function for delta update (e.g., Oja/Sanger)
- `recalibrate_fn`: Orthonormalization method
- Plus data, learning rate, stopping rule, etc.

### ✴ Learning Rates (in `learning_rate_schemas.py`)
Functions receiving only `step` and returning the scalar learning rate.

### ✴ Sample Generation (in `utilities.py`)
- `generate_samples(num_samples, rng=None)`  
Returns matrix of shape `(d, num_samples)`, or 1D vector if `num_samples == 1`.

### ✴ Stopping Rule (in `utilities.py`)
Receives: `delta`, `epsilon`, and optionally `min_iterations`.

---

## 📊 Visualization Functions

- `plot_generalization_errors_trajectory(...)`  
- `plot_angle_trajectories(...)`  
- `plot_samples_components(...)` — **Only for 3D**

To use in higher dimensions, skip/remove `plot_samples_components` from `assignment_plotting` (function inside assignment_code.py).

---

## 🧪 Debug Outputs

When `--debug` is enabled:
- All eigenvectors of correlation matrix (not just top-k)
- Found components
- Mean of generated samples
- Best achievable generalization error on validation set

---

## 🔧 Custom Rules

Create your own PCA rule:
1. Add a `.py` file to `rules/`
2. Implement:
   - `delta_fn(sample, components)`
   - `recalibrate_components(W)`
   - `run_your_rule_online_pca(...)` → calls `run_online_pca(...)` with your functions

---

## 📄 License

MIT License. 
See `LICENSE` for details.

```

