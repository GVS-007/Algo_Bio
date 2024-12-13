# CSE 559: Algorithms in Computational Biology Final Project

**Authors:** Venkata Sai Gunda (1229553808), Akhil Routhu (1229582668),  
Ramesh Babu Mannam (1230786364), Siva Kumar Katta (1232008975),  
Vijetha Kasala (1231930780)

## Introduction

Predicting which T-cell receptors (TCRs) bind to which epitopes (peptide fragments) is crucial in understanding immune responses and developing effective immunotherapies and vaccines. In this project, we leverage TCR-BERT and catELMo embeddings of TCR and epitope sequences to build machine learning models that predict TCR-epitope binding affinity.

We experimented with four main models:

1. **Baseline TCRBert Model**:  
   A baseline neural network that uses per-sequence (CLS token) embeddings for both TCRs and epitopes from TCR-BERT. These embeddings are concatenated and passed through a fully connected network.

2. **Cross-Attention TCRBert Model**:  
   A more complex model that uses per-token embeddings from TCR-BERT and applies bi-directional cross-attention. The intuition is that the TCR queries the epitope and the epitope queries the TCR, potentially revealing intricate relationship patterns.

3. **Baseline CatELMo LSTM Model**:  
   A bidirectional LSTM-based model fine-tuned to generate embeddings for TCR and epitope sequences.

4. **CatELMo-Transformer Model**:  
   A modified version of the CatELMo model that replaces the LSTM layers with transformer encoders to better capture token-level interactions.

## Experimental Setup

**Data:**  
We used the `TCREpitopePairs.csv` dataset, which contains TCR sequences, epitope sequences, and binary binding labels.

**Embedding Generation:**
- Baseline TCRBert Model: Extracted CLS token embeddings for TCR & epitopes using TCR-BERT.
- Cross-Attention TCRBert Model: Extracted per-token embeddings for TCR & epitope, allowing attention across all token positions.
- Baseline catELMo Model: Generated embeddings using bidirectional LSTMs fine-tuned specifically for TCR and epitope sequences.
- catELMo-Transformer Model: Leveraged transformer encoders to generate embeddings with hyperparameters tuned for token-level interactions.

**Data Splits:**
- **Random Split**: Train-test split done randomly.
- **TCR Split**: Ensures test TCRs do not appear in training.
- **EPI Split**: Ensures test epitopes do not appear in training.

**Metrics:**  
We computed Accuracy, Precision, Recall, F1-Score, and AUC over 5 runs and averaged the results for stability.

**Hyperparameters:**

- **Baseline TCR-Bert Model**:
  - Activation: ReLU
  - Dropout: 0.5
  - Learning Rate: 1e-4
  - Hidden Layers: [1024, 512, 256]
  - Max Sequence Length (for embeddings): 50 tokens
  - Batch Size: 32
  - Epochs: 10

- **Cross-Attention TCR-Bert Model**:
  - Heads: 4
  - Linear Layers: [1024, 512]
  - Dropout: 0.25
  - Learning Rate: 1e-4
  - Max Sequence Length: 3
  - Batch Size: 16
  - Epochs: 10

- **Baseline catELMo LSTM Model and catELMo-Transformer Model** (not the main focus of these final results, but previously tested):
  - LSTM Layers: 4 (catELMo)
  - Transformer Layers: 6 (catELMo-Transformer)
  - Hidden State Dim: 4096 (catELMo)
  - Heads: 8 (catELMo-Transformer)
  - Feed-Forward Dim: 2048 (catELMo-Transformer)
  - Dropout: 0.1
  - Batch Size: 128

## Results

### Baseline TCR-Bert Model Results (Averaged Over 5 Runs)

| Split  | Precision | Recall   | F1-Score | Accuracy | AUC     |
|--------|-----------|----------|----------|----------|---------|
| tcr    | 0.770920  | 0.626009 | 0.690371 | 0.720695 | 0.801577|
| epi    | 0.726388  | 0.562728 | 0.633745 | 0.675040 | 0.739110|
| random | 0.766363  | 0.632150 | 0.692210 | 0.719402 | 0.800171|

**Observations:**  
The baseline model provides stable and balanced performance across splits. The TCR and EPI splits are out-of-distribution scenarios, yet the model maintains reasonable F1-scores (around 0.63 to 0.69) and AUC (0.74 to 0.80).

### Cross-Attention TCR-Bert Model Results (Averaged Over 5 Runs)

| Split  | Precision | Recall   | F1-Score | Accuracy | AUC     |
|--------|---------- |----------|----------|----------|---------|
| tcr    | 0.976473  | 0.074132 | 0.137801 | 0.538081 | 0.559846|
| epi    | 0.853305  | 0.014510 | 0.028534 | 0.506005 | 0.504402|
| random | 0.917487  | 0.083598 | 0.151402 | 0.535861 | 0.558465|


## Repository Structure

```bash
.
├── base_model.py           # Code for the base model
├── cross_attention.py      # Code for the cross-attention model
├── run.sh                  # SLURM script to run models on HPC
├── pytorch_a100_env.yml    # Conda environment file
├── models/
│   ├── best_model_random.pt         # Best model weights for a particular config
│   ├── best_model_tcr.pt
│   ├── best_model_epi.pt
│   ├── base_best_model_random.pt
│   ├── ... (other saved models)
└── data/
    └── TCREpitopePairs.csv # The dataset (not included in repo, must be provided)
```
## Setup

### Conda Environment

Create and activate the conda environment:

```bash
conda env create -f pytorch_a100_env.yml
conda activate pytorch_a100_env
```

Ensure that the environment has all the required dependencies.

## Running Experiments

### Data Preperation

- Place TCREpitopePairs.csv in the data/ directory.
- Ensure the dataset has tcr, epi, and binding columns.

Running the Base Model

```bash
python base_model.py
```

This script:
- Generates embeddings for TCR and epitope sequences.
- Trains the base model.
- Saves model weights to models/base_best_model_<split>.pt.
- Outputs evaluation metrics.


Running the Cross-Attention Model

```bash
python cross_attention.py
```

This script:
- Generates per-token embeddings.
- Trains the cross-attention model.
- Saves weights to models/best_model_<split>_len<max_length>.pt.
- Outputs evaluation metrics for various splits and sequence lengths.



## Using run.sh with SLURM

For HPC environments:

```bash
sbatch run.sh
```
- Modify run.sh to request the appropriate resources (CPU, GPU, memory).
- The script will run the model code using SLURM scheduling.

## Saving and Loading Models

The best models are saved in models/.

To load a model:
```bash
from base_model import DeepNN  # or from cross_attention import BiDirectionalCrossAttentionNet
import torch

model = DeepNN(input_size, num_classes)
model.load_state_dict(torch.load('models/base_best_model_random.pt'))
model.eval()
```
Adjust paths and architectures as needed.

## Acknowledgements
- TCR-BERT: wukevin/tcr-bert
- PyTorch: https://pytorch.org/
- Transformers Library: https://huggingface.co/transformers/
