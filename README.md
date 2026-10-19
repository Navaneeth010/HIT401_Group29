# Cross-Domain Fake News Detection Using Deep Learning: An Anomaly-Based Approach

Code for the HIT401 Capstone Project (Charles Darwin University, Group 29). Each fake news detector is trained on one domain and tested on that domain and on unseen domains.

## Data

The datasets are not included. Download them from their sources:

| Dataset | Source |
|---|---|
| ISOT (Ahmed et al. 2017) | https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/ |
| LIAR (Wang 2017) | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip |
| FakeNewsNet (Shu et al. 2020) | https://github.com/KaiDMML/FakeNewsNet (CSV files in `dataset/`) |

Place the files as follows:

```text
data/
├── ISOT/          Fake.csv, True.csv
├── LIAR/          train.tsv, valid.tsv, test.tsv
└── FakeNewsNet/   politifact_fake.csv, politifact_real.csv, gossipcop_fake.csv, gossipcop_real.csv
```

## Environment

- **Python:** 3.11 (the reported runs used 3.11.9).
- **Libraries** (pinned in `requirements.txt`):
  - PyTorch 2.8.0 (CUDA 12.8 build)
  - Hugging Face Transformers 5.17.0
  - scikit-learn 1.9.0
  - NumPy 2.1.3
  - pandas 3.0.3
- **Hardware:** the reported runs used the free tier of Google Colab with one NVIDIA T4 GPU (16 GB). A CUDA GPU is strongly recommended for the Transformer models.
- **Downloads on first run:** the CNN and BiLSTM need the Google News word2vec vectors (1.7 GB). These are downloaded automatically, or you can supply an existing file with `--w2v-path`.

## Installation

```bash
pip install -r requirements.txt
```

If pip installs a CPU-only PyTorch, install the CUDA build instead:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### Google Colab

1. Choose **Runtime → Change runtime type → T4 GPU**.
2. Mount Google Drive, copy the code there and put the datasets in `data/` next to it.
3. Install the requirements and run the experiments, writing the results to Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/<repository folder>
!pip install -r requirements.txt
!python run_experiments.py --data-dir data --out results
```

Free-tier sessions disconnect after a few hours. Run the same command again in a new session: experiments that are already in `results/metrics.csv` are skipped, so the run continues where it stopped.

## Parameter settings

All settings are fixed in the code.

| Model (key) | Settings |
|---|---|
| TF-IDF + logistic regression (`LR`) | unigrams + bigrams, 30,000 features, English stop words; C = 1 |
| Text CNN (`CNN`) | word2vec 300-d embeddings (fine-tuned); filter widths 3/4/5 with 100 filters each; dropout 0.5; Adam, learning rate 0.001, batch 64, up to 10 epochs, early stopping with patience 2 |
| BiLSTM (`BiLSTM`) | same embeddings; one layer with 128 units per direction; dropout 0.5; same optimiser and stopping as the CNN |
| DistilBERT, BERT, RoBERTa (`DistilBERT`, `BERT`, `RoBERTa`) | `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`; AdamW, learning rate 2e-5, weight decay 0.01, batch 16, 3 epochs, 10% warm-up |
| RoBERTa probe (`Probe`) | frozen `roberta-base`, mean-pooled, PCA to 128 dimensions; logistic regression, C = 1 |
| One-class detector (`OneClass`) | same features as the probe; squared Mahalanobis distance from genuine news (Ledoit–Wolf covariance); threshold τ = 90th percentile of genuine source-validation scores |

Settings shared by all models:
- **Input length:** 400 word tokens (CNN, BiLSTM) or 256 subword tokens (Transformers) for ISOT, and 64 for the other datasets.
- **Splits:**
  - ISOT: stratified 75/25 train/test, with 10% of the training part used for validation.
  - LIAR: the official files.
  - PolitiFact and GossipCop: stratified 70/10/20.
- **Seed:** 42.
- **Class weights:** when training on GossipCop, the fake class is weighted by the genuine/fake ratio.
- **Model selection:** on the source validation split only.

| Setting | Trained on | Tested on |
|---|---|---|
| S1 | ISOT | ISOT |
| S2 | ISOT | LIAR |
| S3 | LIAR | LIAR |
| S4 | PolitiFact | PolitiFact |
| S5 | GossipCop | PolitiFact |
| S6 | GossipCop | GossipCop |
| S7 | PolitiFact | GossipCop |

## Running the experiments

Run from the folder that contains `run_experiments.py`. This command runs all eight models on all seven settings:

```bash
python run_experiments.py --data-dir data
```

| Option | Default | Meaning |
|---|---|---|
| `--data-dir` | `data` | folder containing `ISOT/`, `LIAR/` and `FakeNewsNet/` |
| `--out` | `results` | output folder |
| `--models` | all | any of `LR CNN BiLSTM DistilBERT BERT RoBERTa Probe OneClass` |
| `--settings` | all | any of `S1` … `S7` |
| `--w2v-path` | download | local word2vec file |
| `--rerun` | off | repeat experiments that are already in `metrics.csv` |
| `--max-items` | off | subsample every split for a quick functional test |

## Outputs

| File | Content |
|---|---|
| `results/metrics.csv` | one row per experiment: confusion counts, accuracy, precision, recall, F1 and balanced accuracy |
| `results/threshold_sweep.csv` | one-class detector results across thresholds |
| `results/run_log.txt` | library versions, GPU, split sizes and training progress |
