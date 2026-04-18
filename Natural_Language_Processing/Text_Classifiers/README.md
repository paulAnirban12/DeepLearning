# GRU and BERT for NLP

This project contains two notebook implementations for natural language processing tasks:

- **GRU_Text_Classifier.ipynb** — a GRU-based text classification model trained on the IMDB movie reviews dataset
- **BERT_Text_Classifier.ipynb** — a BERT-based Named Entity Recognition (NER) model fine-tuned on the CoNLL-2003 dataset

---

## Files Included

- `GRU_Text_Classifier.ipynb`
- `BERT_Text_Classifier.ipynb`
- `README.md`

---

## Project Overview

### 1. GRU Text Classifier
This notebook implements a simple GRU model in **PyTorch** for text classification.

It includes:
- loading the IMDB dataset
- text preprocessing and tokenization
- vocabulary creation
- padding/truncation of sequences
- GRU model training
- validation and test evaluation
- sample predictions on unseen reviews

### 2. BERT Text Classifier
This notebook implements a BERT-based NER model using **Hugging Face Transformers** and **PyTorch**.

It includes:
- loading the CoNLL-2003 dataset
- tokenization and label alignment
- fine-tuning a pretrained BERT model
- evaluation using precision, recall, F1-score, and accuracy
- sample predictions on unseen test sentences

