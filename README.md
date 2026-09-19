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

## Source and license

The underlying sentences come from the **Vashantor** dataset by Faria et al.

- Paper: https://arxiv.org/abs/2311.11142
- Data: https://data.mendeley.com/datasets/bj5jgk878b/2
- Repository: https://github.com/Mukaffi28/Vashantor-A-Large-scale-Multilingual-Benchmark-Dataset

Vashantor is licensed under **CC BY 4.0**
(https://creativecommons.org/licenses/by/4.0/).

**Changes made:** the per-dialect files were merged into a single parallel
corpus, with one row per sentence and one column per variety, then re-split
into train / validation / test. Sentence content was not altered.

The files in this folder are a modified version of Vashantor and are therefore
also distributed under CC BY 4.0. Please cite the original authors:
...
@article{faria2023vashantor,
  title={Vashantor: a large-scale multilingual benchmark dataset for automated translation of bangla regional dialects to bangla language},
  author={Faria, Fatema Tuj Johora and Moin, Mukaffi Bin and Wase, Ahmed Al and Ahmmed, Mehidi and Sani, Md Rabius and Muhammad, Tashreef},
  journal={arXiv preprint arXiv:2311.11142},
  year={2023}
}
...

## Environment

Two environments were used, since the models were trained in different places.


### Local — dialect identification and full pipeline

Conda environment, Python 3.10, on Windows with CUDA.

```bash
conda create -n banglabert python=3.10 -y
conda activate banglabert

# PyTorch — pick the command matching your CUDA version from pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.37.2 tokenizers==0.15.2 accelerate==0.30.0
pip install datasets evaluate sacrebleu nltk rouge-score sentencepiece
pip install protobuf==3.20.3
pip install jupyter
```

Then launch Jupyter from inside the environment:

```bash
jupyter notebook
```

If the notebooks use NLTK metrics, download the required corpora once:

```python
import nltk
nltk.download("punkt")
nltk.download("wordnet")
```

**Version note:** the saved checkpoints were written with transformers
**4.57.3**, while the environment above pins **4.37.2**. Loading the
checkpoints under the older version raises config errors — upgrade
transformers to match the checkpoints, or re-save them under 4.37.2.


### Kaggle — translation

Kaggle's default GPU image (T4), plus:

```
protobuf 3.20.3
portalocker 2.8.2
rapidfuzz 3.5.2
jiwer 3.0.1
sacrebleu 2.4.0
```


## Citation

```
@inproceedings{israt2026unified,
  title={Unified Bangla: A Transformer-Based Framework for Bangla Dialect Identification and Cross-Dialectal Translation},
  author={Israt, Tasfia and Anannya, Mehrin and Mahfuz, Sadia and Shourov, Riad Mashrub and Hosen, Md Biplob and Mazumder, Rashed},
  booktitle={2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence \& Networking (QPAIN)},
  pages={1--6},
  year={2026},
  organization={IEEE}
}
```
