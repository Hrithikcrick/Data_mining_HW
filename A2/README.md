# COL761 Assignment 2

This repository contains solutions for **COL761 Assignment 2 (Spring 2026)**.

---

## Kerberos ID Used

**jtm252082**

---

## Repository Structure

```
A2
├── env.sh
├── q1
│   ├── Q1.py
│   ├── plot.png
│   └── report_q1.pdf
└── q2
    ├── forest_fire.sh
    ├── fireblock.py
    └── report_q2.pdf
```

---

## Environment Setup

Run the following command to install the required Python packages:

```bash
bash env.sh
```

---

## Running Q1 (K-Means Clustering)

### Using dataset from API

```bash
python3 Q1.py 1
```

or

```bash
python3 Q1.py 2
```

### Using local dataset

```bash
python3 Q1.py dataset.npy
```

### Output

* `plot.png` — Visualization of clustering results
* Optimal **k** printed to stdout

---

## Running Q2 (Forest Fire Blocking)

```bash
bash forest_fire.sh <graph> <seedset> <output_file> <k> <r> <hops>
```

### Example

```bash
bash forest_fire.sh dataset1/dataset_1.txt dataset1/seedset_1.txt out.txt 50 50 -1
```

### Output

The output file contains **exactly `k` blocked edges**.

---

## Reports

* `q1/report_q1.pdf` — Clustering analysis and results
* `q2/report_q2.pdf` — Route blocking formulation and algorithm
