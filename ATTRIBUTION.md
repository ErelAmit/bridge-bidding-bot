# Attribution

This document lists external resources, libraries, datasets, and AI assistance used in this project.

---

## Datasets and Problem Setup

- The project uses six `.mat` files that contain:
  - Feature matrices with 104 dimensional binary vectors per deal (two 52 card hands for North and South).
  - IMP cost matrices with 36 dimensional vectors of IMP losses for each candidate bid.
- The representation and task follow prior research on contract bridge bidding that uses:
  - A fixed ordering of the 52 cards to encode each hand as a boolean vector.
  - A discrete action space of bidding actions and an IMP based cost function for each action.
- The `.mat` files themselves were provided from existing bridge bidding research by Chih-Kuan Yeh and Hsuan-Tien Lin; the git repository from where I pulled the data can be found at: https://github.com/chihkuanyeh/Automatic-Bridge-Bidding-by-Deep-Reinforcement-Learning.git

---

## External Libraries

The project relies on the following external libraries:

- **PyTorch** for tensor operations, model definition, training loops, and evaluation  
  - Used in `src/models/*.py`, `src/train_supervised.py`, `src/train_imps_hybrid.py`, and all evaluation scripts.
- **NumPy** for basic numerical operations and array handling.
- **SciPy** for loading MATLAB `.mat` files via `scipy.io.loadmat`.
- **Matplotlib** for plotting training and evaluation curves into `.png` files under `docs/`.

All of these libraries are used in standard ways according to their documentation.

---


## AI Generated Code and Assistance

AI tools (specifically ChatGPT) were used as a coding assistant in the following ways:

- Drafting initial versions of core modules and scripts, including:
  - `src/data/bridge_dataset.py` (dataset loading, train and test splits, IMP dataset wrapper)
  - `src/models/mlp_bidder.py` (MLP architecture with batch normalization and dropout)
  - `src/models/attn_rnn_bidder.py` (suit based tokenization and attention plus RNN architecture)
  - Training scripts such as `src/train_supervised.py` and `src/train_imps_hybrid.py`
  - Evaluation utilities such as `src/eval_train_val.py`, `src/eval_test_mlp.py`, `src/eval_imps_mlp.py`, `src/eval_imps_attn_rnn.py`
  - Plotting utilities such as `src/plot_training_curves.py` and `src/plot_attn_rnn_imps_curves.py`
  - Comparison script `src/compare_mlp_vs_attn_rnn.py` for side by side IMP metrics
  - The Colab pipeline notebook `notebooks/bridge_bidding_full_pipeline.ipynb`
- Refactoring and improving code organization:
  - Breaking monolithic code into reusable functions and classes
  - Aligning training and evaluation code with PyTorch best practices
  - Adding command line entry points using `python -m src.<module>`
- Drafting documentation and supporting files:
  - Initial versions of `README.md`, `SETUP.md`, and `ATTRIBUTION.md`
  - Guidance text and explanations for evaluation and results
- Proper usage and sytax for:
    - git
    - github
    - anaconda prompt

All AI generated code and text was reviewed, edited, and integrated by the student. Logical choices such as model hyperparameters, architecture variants, and evaluation design were made by the student, not by copying any existing project directly.


