Dataset Source: https://drive.google.com/drive/folders/16lHceA9Y0e3BMD6Ru3sOl7PP7yO2_s50?usp=drive_link

# Part 3: NLP and Sequence Modeling Mini Project

## Project Overview

This project builds a basic NLP pipeline for a customer support text classification problem. The goal is to classify customer messages into three sentiment classes: **positive**, **neutral**, and **negative**.

The project compares two approaches:

1. **Traditional text vectorization** using TF-IDF with Logistic Regression.
2. **Sequence-based deep learning** using tokenized padded sequences with a GRU model.

## Dataset

The dataset used is `customer_support_text_classification.csv`.

Important columns:

- `customer_message`: input text used for prediction
- `sentiment_label`: target label with three classes — positive, neutral, and negative
- `channel`: source channel of the support ticket
- `word_count`: derived text length indicator
- `urgent_flag`: optional indicator showing urgency

The model mainly uses `customer_message` as the input and `sentiment_label` as the target.

## Repository Structure

```text
part-3-nlp-sequence-modeling/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── data/
│   ├── customer_support_text_classification.csv
│   └── data_dictionary.md
└── results/
    ├── model_evaluation.png
    └── sample_predictions.txt
```

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Then open and run:

```text
notebook.ipynb
```

The notebook contains the task-by-task work, including the Task 6 reflection.

## Model Summary

### Baseline Model

The baseline model uses:

- cleaned customer messages
- TF-IDF vectorization
- Logistic Regression classifier

This gives a strong simple benchmark because TF-IDF captures important keywords and phrases that are useful for sentiment classification.

### Sequence Model

The sequence model uses:

- tokenized text
- padded/truncated input sequences
- embedding layer
- GRU sequence layer
- dense output layer with three classes

This model is useful because it treats text as an ordered sequence, instead of only a bag of words.

## Output Files

The required outputs are saved in the `results/` folder:

- `model_evaluation.png`: comparison of baseline and sequence model metrics
- `sample_predictions.txt`: sample predictions from the trained baseline model
