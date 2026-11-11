# Memory-Enhanced Graph Convolutional Neural Network (MEGCNN) for MOF Electrocatalyst Screening

This repository contains the core implementation of MEGCNN, as described in our manuscript "Accelerated Discovery of Bifunctional MOF Electrocatalysts via Memory-Enhanced Graph Neural Networks".

## 📂 Repository Structure
- `main.py`: The complete code for data loading, crystal graph construction, MEGCNN model, and 5-Fold/LOMO validation.
- `data/sample_data.xlsx`: A **minimal example** containing 3 MOF samples (descriptors + target adsorption energies).
- `relaxed_structures/`: The corresponding CIF files for the 3 sample MOFs.
- `requirements.txt`: Required Python libraries.

## ⚙️ Quick Start

1. Install dependencies:
   `pip install -r requirements.txt`

2. Run the example:
   `python main.py`

**Important Note:** Because this public repository only contains a 3-sample subset for demonstration purposes, the training loop and cross-validation will not complete successfully. However, the script clearly demonstrates:
  1. How 3D MOFs are parsed into crystal graphs (node/edge embeddings).
  2. The implementation of the GNN message-passing layers and the External Memory Bank.
  3. The dual-input fusion mechanism.

For access to the full 293-MOF dataset and the trained weights, please refer to the manuscript's Supporting Information or contact the corresponding author.