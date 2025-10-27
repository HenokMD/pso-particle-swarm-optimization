# Particle Swarm Optimization (PSO) – Benchmark Suite

Reproducible Python implementations of **single-objective** and **multi-objective** Particle Swarm Optimization:
- Rosenbrock (unimodal)
- Belledonne (multimodal)
- 2-objective benchmark with an **MPSO** (Pareto) variant

##  Components

```
pso-particle-swarm-optimization/
├─ src/
│  ├─ Rosenbrock_PSO_Evaluation.py
│  ├─ Belledone_PSO_Evaluation.py
│  └─ MULTI_OBJECTIVE_PSO.py
├─ .gitignore
├─ LICENSE
├─ requirements.txt
└─ README.md
```

## Quick start

```bash
# 1) Create & activate a virtual env (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run any script
python src/Rosenbrock_PSO_Evaluation.py
python src/Belledone_PSO_Evaluation.py
python src/MULTI_OBJECTIVE_PSO.py
```

Each script prints optimization progress and shows the corresponding plots.

## 🧪 What’s included

- **`Rosenbrock_PSO_Evaluation.py`**: PSO with early stopping + sensitivity/convergence plots.
- **`Belledone_PSO_Evaluation.py`**: Improved PSO with random inertia, velocity clamping, and multi-run restarts.
- **`MULTI_OBJECTIVE_PSO.py`**: Minimal **MPSO** implementation with archive, crowding distance, and binary-tournament leader selection.

## ✅ Python version

Python 3.9+ is recommended.

## 📝 Citation

If you use this code or the accompanying report, please consider citing this repository.

## 📄 License

MIT — see `LICENSE`.
