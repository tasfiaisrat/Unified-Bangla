# Unified Bangla

Code for **Unified Bangla: A Transformer-Based Framework for Bangla Dialect Identification and Cross-Dialectal Translation**,  2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence & Networking (QPAIN), 2026.

A transformer-based framework for Bangla regional dialect identification
and cross-dialectal translation on text, covering five regional dialects —
Barishal, Chittagong, Mymensingh, Noakhali and Sylhet — plus Standard Bangla.

## Pipeline

1. **Dialect identification** — BanglaBERT, compared against DistilBERT.
   BanglaBERT performed better and is the model used in the final pipeline.
2. **Cross-dialectal translation** — mBART-50, fine-tuned with the target
   dialect tag prefixed to each source sentence.
3. **End-to-end pipeline** — identification feeding translation.


## Notebooks

| Notebook | Stage | Trained on |
|---|---|---|
| `notebooks/01_dialect_identification.ipynb` | BanglaBERT + DistilBERT | Local (conda, CUDA) |
| `notebooks/02_translation_mbart50.ipynb` | mBART-50 fine-tuning | Kaggle (T4 GPU) |
| `notebooks/03_full_pipeline.ipynb` | Aggregated pipeline | Local (conda) |

## Data

Built from the **Vashantor** dataset, which is distributed as separate files
per dialect. Those files were merged into a single **parallel corpus**: each
row holds the same sentence written in Standard Bangla and in each of the
five regional dialects. The corpus is then split into three CSVs:

```
data/
├── Unified_train_dataset.csv
├── Unified_validation_dataset.csv
└── Unified_test_dataset.csv
```

Each CSV has one column per variety:

| Column | Variety |
|---|---|
| `bangla_speech` | Standard Bangla |
| `barishal_bangla_speech` | Barishal |
| `chittagong_bangla_speech` | Chittagong |
| `mymensingh_bangla_speech` | Mymensingh |
| `noakhali_bangla_speech` | Noakhali |
| `sylhet_bangla_speech` | Sylhet |

The parallel structure is what makes all 30 ordered dialect-to-dialect
translation directions available from a single corpus.

The CSVs are included in this repository.

## Environment

Two environments were used, since the models were trained in different places.

### Local — dialect identification and full pipeline

Conda environment, Python 3.10:

```
transformers 4.37.2
accelerate 0.30
tokenizers 0.15.2
datasets
evaluate
sacrebleu
nltk
rouge-score
protobuf 3.20.3
```


### Kaggle — translation

Kaggle's default GPU image (T4), plus:

```
protobuf 3.20.3
portalocker 2.8.2
rapidfuzz 3.5.2
jiwer 3.0.1
sacrebleu 2.4.0
```

## Model weights

Trained checkpoints are too large for GitHub and are hosted separately.
[Add the link here once uploaded.]

