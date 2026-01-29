
# MetaACO-RCPSP

Self-replicating Ant Colony Optimization (ACO) framework for the  
**Resource-Constrained Project Scheduling Problem (RCPSP)**  
with experiments on **PSPLIB** benchmark instances.

This repository contains a research-oriented implementation combining
Ant Colony Optimization, Serial Schedule Generation (SGS),
multi-colony coordination via UCB bandits, and evolutionary
self-replication of solver parameters.

---

## Features

- Ant Colony Optimization with **position-based pheromones**
- **Serial Schedule Generation Scheme (SGS)** for RCPSP
- Multiple colonies with heterogeneous parameters
- **UCB multi-armed bandit** for adaptive colony selection
- Self-replication and mutation of successful colonies
- Elite archive and global pheromone reinforcement
- Support for **PSPLIB** instances (j30 / j60 / j90)

---

## Project Structure
````
metaACO/
├── psplib/                # PSPLIB benchmark instances (.sm)
│   ├── j301_1.sm
│   └── …
└── rcpsp/
├── metaACO.py         # main solver
└── .venv/             # local virtual environment (not committed)
````
---

## Requirements

- macOS or Linux
- Python **3.9+** (tested with Python 3.13)
- PSPLIB `.sm` benchmark files

---

## Installation

### 1. Clone the repository

```bash
git clone 

2. Create a virtual environment

python3 -m venv .venv

3. Activate the environment

source .venv/bin/activate

4. Install dependencies

python -m pip install --upgrade pip
python -m pip install psplib

5. Verify installation

python -c "import psplib; print('psplib OK')"

Expected output:

psplib OK

```
---

Usage

Basic run (PSPLIB j30)

From the rcpsp directory:

```
python metaACO.py ../psplib/j301_1.sm

Run with parameters

python metaACO.py ../psplib/j301_1.sm \
  --seed 1 \
  --iterations 500 \
  --ants 20 \
  --init-colonies 3 \
  --max-colonies 10

Run without activating the virtual environment (alternative)

.venv/bin/python metaACO.py ../psplib/j301_1.sm
```

⸻

Command-Line Arguments
````
Argument	Description	Default
file	Path to PSPLIB .sm file	required
--seed	Random seed	42
--iterations	Number of meta-iterations	500
--ants	Ants per iteration	20
--init-colonies	Initial number of colonies	3
--max-colonies	Maximum number of colonies	10
````
---

Output

Example output:

Best makespan: 73
Best colony params: ColonyParams(alpha=1.21, beta=3.9, rho=0.05, q0=0.68, ...)
Start times: [0, 0, 4, 6, 8, 13, ...]

	•	Best makespan — optimization objective value
	•	Start times — start time of each job (job IDs 0..n−1)

---

Benchmark Notes
	•	j30 — recommended for debugging and algorithm analysis
	•	j60 — reduce number of ants and iterations
	•	j90 — computationally expensive; start with:

--iterations 100 --ants 5



---


Limitations
	•	Single-mode RCPSP only
	•	Renewable resources only
	•	No job preemption
	•	Minimal CLI (no subcommands, no logging flags)

