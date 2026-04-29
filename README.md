# MINLP Surrogate Project

This repository contains two main case studies exploring Generalized Neural MINLP Surrogates and optimization pipelines. The repository has been restructured to house only the essential, functional case studies.

## How to Install Dependencies
Before running the models, install the required dependencies (using Python 3.10+ is recommended):
```bash
pip install -r requirements.txt
```

## 1. Hybrid Vehicle Control Case Study
**Directory:** `Hybrid_Vehicle_Case_Study/`

Contains the experiments and datasets for evaluating differentiable MINLP surrogates on the hybrid electric vehicle control problem. 

**Key Files:**
- `run_hybrid_vehicle.py` / `.ipynb`: Main scripts to generate data, train the surrogate, and execute the complete experimental sweep comparing the Decision-Focused surrogate against Standard NN baselines.
- `minlp_experiment_ready.ipynb`: Additional notebook for interactive experimentation.

**How to Run:**
To execute the full benchmarking sweep (which tests over different constraint violation settings and neural network sizes):
```bash
cd Hybrid_Vehicle_Case_Study
python run_hybrid_vehicle.py
```
*Alternatively, you can open `run_hybrid_vehicle.ipynb` in Jupyter Notebook/Lab to run it interactively and visualize the plots step-by-step.*

## 2. ACOPF UC Case Study
**Directory:** `ACOPF_UC_Case_Study/`

Contains the models, source code, and data for exploring ACOPF (AC Optimal Power Flow) Unit Commitment case studies using different levels of supervision.

**Key Files:**
- `run_acopf_unsupervised.py` / `.ipynb`: Unsupervised decision-focused model implementation.
- `run_acopf_binary.py`: Model handling direct binary prediction setups.
- `src/`: Modular code containing the PyTorch model architectures (`model.py`, `model_binary.py`), CVXPY differentiable layers (`cvxpy_layer.py`), and physical formulations.
- `data/`: ACOPF use cases (e.g., `case14.m`, `case300.m`).

**How to Run:**
To run the unsupervised differentiable optimization pipeline:
```bash
cd ACOPF_UC_Case_Study
python run_acopf_unsupervised.py
```
To run the binary variant of the surrogate model:
```bash
cd ACOPF_UC_Case_Study
python run_acopf_binary.py
```
*You can also explore `run_acopf_unsupervised.ipynb` or `DF_model_5.ipynb` via Jupyter for detailed cell-by-cell analysis and outputs.*
