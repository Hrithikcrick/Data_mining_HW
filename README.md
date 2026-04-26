# Data_mining_HW
# COL761 Assignment 3 Submission  


## Overview

This repository contains my submission for **COL761 Assignment 3**. The assignment has two parts:

1. **Q1: Search in High Dimensional Space**
2. **Q2: Prediction on Graph-Structured Data**

The submission follows the required directory structure and uses the assignment-specified interface and scripts.

---

## Directory Structure

```text
jtm252082/
├── Q1/
│   └── submission.py
├── Q2/
│   └── src/
│       ├── load_dataset.py
│       ├── predict.py
│       ├── evaluate.py
│       ├── train.py
│       ├── train_A.py
│       ├── train_B.py
│       ├── train_C.py
│       ├── models.py
│       └── utils.py
└── requirements.txt

## Environment Used

### Assignment-required environment
- **Python:** 3.10
- **PyTorch:** 2.7.1
- **PyTorch-Geometric:** 2.7.0
- Other packages are listed in `requirements.txt`.

### Local validated environment used

For Linux/WSL validation of Q2, I used:
- **Python:** 3.10.20
- **Torch:** 2.7.1+cu118
- **PyG:** 2.7.0
- **NumPy:** 2.2.6
- **scikit-learn:** 1.7.2
- **tqdm:** 4.67.3

For Q1 local testing, I also used a Windows virtual environment with:
- `numpy`
- `faiss-cpu`
