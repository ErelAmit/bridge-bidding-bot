# Project Setup and Installation

This document explains how to set up the environment and run the bridge bidding project, both locally and in Google Colab.

You can either:

- Run everything in Google Colab with GPU (recommended for training speed), or
- Run locally using a conda environment and CPU.

The project does not require any external APIs or services. 

---

## 1. Project structure

At the top level, the repository is organized as follows:

- `src/` source code for data loading, models, training, evaluation, and plotting  
- `data/` data files (`.mat`) and data access scripts  
- `models/` trained model checkpoints and model loading scripts  
- `notebooks/` Jupyter and Colab notebooks used for exploration or full pipelines  
- `videos/` demo and technical walkthrough videos (to be added)  
- `docs/` plots, metrics CSVs, and other documentation artifacts  
- `README.md` main project overview and usage guide  
- `SETUP.md` this setup guide  
- `ATTRIBUTION.md` attribution of AI assistance, external code, and datasets  

---

## 2. Running in Google Colab (recommended)

The simplest way for graders to reproduce the main results is via Google Colab with GPU.

1. Open the notebook:

   - `notebooks/bridge_bidding_full_pipeline.ipynb`

2. In Google Colab:

   - Go to `Runtime` → `Change runtime type`  
   - Set `Hardware accelerator` to `GPU`  
   - Click `Save`

3. Run the first cell in the notebook. It will:

   - Clone this repository under `/content/bridge-bidding-bot`  
   - Install minimal dependencies (`scipy`, `matplotlib`)

4. Run all remaining notebook cells in order. The notebook will:

   - Train the supervised MLP baseline  
   - Plot training curves for loss and accuracy  
   - Run a small hyperparameter and optimizer sweep  
   - Train the hybrid attention plus RNN model using IMP based loss  
   - Plot hybrid training curves  
   - Evaluate all models and baselines on IMP metrics  
   - Print a comparison table with random, majority, MLP, and hybrid performance

---

## 3. Local setup with conda (CPU)

If you prefer to run the project locally:

1. From a terminal or Anaconda Prompt:

   - conda create -n bridge-bot python=3.11 -y
   - conda activate bridge-bot

2. Clone the repository:

   - git clone https://github.com/ErelAmit/bridge-bidding-bot.git
   - cd bridge-bidding-bot


3. Install Python dependencies:

   - pip install torch scipy matplotlib

4. Run training and evaluation scripts as desired:
   
      4.1 First, make sure to activate your virtual environment in the right place:

         - conda activate bridge-bot
         - cd path/to/bridge-bidding-bot

      4.2 Then, you can run the MLP baseline:

         - python -m src.train_supervised
         - python -m src.eval_train_val
         - python -m src.eval_test_mlp
         - python -m src.plot_training_curves``

      4.3 Hyperparameter and optimizer sweep:
      
         - python -m src.run_sweep

      4.4 Hybrid attention plus RNN model with IMP based training:

         - python -m src.train_imps_hybrid
         - python -m src.eval_imps_attn_rnn

      4.5 Evaluate MLP model on the same (IMP) metrics:
      
         - python -m src.eval_imps_mlp

      4.6 Compare all models and baselines on IMP metrics:

         - python -m src.compare_mlp_vs_attn_rnn
         