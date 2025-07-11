# Sentiment Analysis with DistilBERT

Model Repository: efarooqi/sentiment_analysis

## Project Task
This is a sentiment analysis tool designed to interpret IMDb movie reviews and classify them as either **Positive (1)** or **Negative (0)**. It uses both traditional machine learning and large language models (LLMs), specifically a fine-tuned and optimized version of `distilbert-base-uncased`.

- Train a Logistic Regression baseline using TF-IDF vectors
- Fine-tune `distilbert-base-uncased` for sequence classification
- Evaluate performance using multiple metrics
- Package and deploy the optimized model via Hugging Face Hub
- Build an interactive demo using Gradio

## Dataset
- **Source**: [IMDb Dataset from Hugging Face Datasets](https://huggingface.co/datasets/imdb)
- **Classes**: Binary classification – `Positive` (1) and `Negative` (0) sentiment labels
- **Size**: Sampled subset of 1,000 reviews for lightweight training
- **Preprocessing**:
  - HTML tag removal
  - Punctuation stripping
  - Lowercasing
  - Tokenization via Hugging Face tokenizer

## Pre-trained Model
- `distilbert-base-uncased` from Hugging Face Transformers
- Fine-tuned using PyTorch and Hugging Face Trainer API
- Label mapping adjusted:
  ```python
  model.config.id2label = {0: "Negative", 1: "Positive"}
  model.config.label2id = {"Negative": 0, "Positive": 1}

## Hyperparameters
A subset of the dataset is tokenized and split into train/test sets. Custom hyperparameters are applied to optimize model performance using Hugging Face's Trainer API. Accuracy and F1 score are used for evaluation


## Performance Metrics
Pre trained model:

### Model Performance

| **Label**        | **Precision** | **Recall** | **F1-Score** | **Support** |
|------------------|---------------|------------|--------------|-------------|
| Negative (0)     | 0.90          | 0.87       | 0.88         | 12,500      |
| Positive (1)     | 0.87          | 0.90       | 0.89         | 12,500      |
| **Accuracy**     | —             | —          | **0.89**     | 25,000      |
| **Macro Avg**    | 0.89          | 0.89       | 0.89         | 25,000      |
| **Weighted Avg** | 0.89          | 0.89       | 0.89         | 25,000      |

> Final Accuracy Score: **88.6%**


![alt text](image-1.png)

After hypertuning

###  Evaluation Metrics (Epoch 4)


| **Metric**                | **Value**     |
|---------------------------|---------------|
| Evaluation Loss           | 0.0351        |
| Evaluation Accuracy       | 99.5%         |
| Evaluation F1 Score       | 0.9943        |
| Evaluation Runtime        | 5.34 seconds  |
| Samples per Second        | 37.46         |
| Steps per Second          | 9.36          |
| Epoch                     | 4             |





After hyperparameter tuning, model locks into high accuracy and F1 with almost negligible training loss, suggesting a powerful fit on training data.

## Final Classification Report Summary

The final model demonstrates strong performance on balanced test samples, with minimal false positives and negatives. Example predictions include:

"Awesome and amazing movie" → Positive

"Terrible acting and dull script" → Negative

### Deployment
Model Repository: efarooqi/sentiment_analysis

Hosted via: Hugging Face Hub

Web Demo: Built with Gradio



