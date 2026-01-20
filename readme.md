# HMI-Pred: Host-Microbe Interaction Predictor

**HMI-Pred** is a novel prediction framework designed specifically for cross-species signaling communication. Unlike general Protein-Protein Interaction (PPI) tools, HMI-Pred is engineered to capture the unique characteristics of microbial ligands and host receptors.

## Key Innovations

The framework incorporates three primary innovations to ensure high accuracy in cross-species prediction:

1.  **Protein Language Model Integration:** Leverages the **ESM-1v** model to extract rich sequence features, enabling the screening of diverse ligands without reliance on crystal structures.
2.  **Dual-Path Framework:** Integrates sequence semantics with biological mechanism constraints for robust cross-validation.
3.  **Specialized Fine-Tuning:** Utilizes the high-quality **BASEHIT** interaction dataset to fine-tune and optimize the prediction of bacteria–immune protein binding interfaces.

---

## Modules and Usage

HMI-Pred consists of three distinct modules designed to handle classification, sequence scoring, and structural docking.

### 1. HMI-Pred-C (Classification Module)

**Purpose:** Screens microbial samples to identify potential ligands that interact with human proteins.

**Usage:**
Run the following command to predict whether a new microbial protein sequence is a potential ligand for human receptors:

```bash
python HMI-Pred-C_predict.py
```
### 2. HMI-Pred-S (Sequence-based Scoring Module)

**Purpose:** Evaluates the interaction probability between human hosts and microbes at the sequence level.

**Description:**
This module leverages the **ESM-1v** protein language model to extract high-dimensional semantic information from sequences. It constructs 1280-dimensional feature vectors for both ligands and receptors to perform precise interaction scoring.

**Usage:**
Run the following command to perform sequence-based interaction prediction:

```bash
python HMI-Pred-S_test.py
```
### 3. HMI-Pred-D (Protein-Protein Docking Module)

**Purpose:** Validates interactions through 3D structure prediction and docking simulations.

**Pipeline:**
This module executes a three-stage structural analysis pipeline:

1.  **Structure Prediction:** Generates 3D structures for ligands and receptors using [ESMFold](https://github.com/facebookresearch/esm).
2.  **Docking Simulation:** Performs rigid-body protein-protein docking using [EquiDock](https://github.com/octavian-ganea/equidock_public).
3.  **Quality Assessment:** Evaluates the biological plausibility of the docking conformations using [GDockScore](https://gitlab.com/mcfeemat/gdockscore).