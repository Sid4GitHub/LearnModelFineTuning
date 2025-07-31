# Twitter Sentiment Analysis with Fine-Tuned BERT

This repository contains a fine-tuned BERT-Base-Uncased model for sentiment analysis on Twitter data. It includes notebooks for fine-tuning, evaluation, and inference, as well as "vanilla" versions for comparison with the base model.

## Project Structure

- **`model-bert-base-uncased.ipynb`**: Notebook providing an overview of the BERT-Base-Uncased model and its usage.
- **`evaluate-vanilla.ipynb`**: Notebook for evaluating the base (non-fine-tuned) BERT model.
- **`inference-vanilla.ipynb`**: Notebook for performing inference using the base (non-fine-tuned) BERT model.
- **`fine-tuner.ipynb`**: Notebook for fine-tuning the BERT-Base-Uncased model on a sentiment analysis dataset.
- **`evaluate.ipynb`**: Notebook for evaluating the fine-tuned model on validation datasets.
- **`inference.ipynb`**: Notebook for performing inference using the fine-tuned model.


## Model Details

### Base Model
- **Name**: BERT-Base-Uncased
- **Architecture**: 12-layer bidirectional Transformer encoder
- **Parameters**: 110 million
- **Pretraining Tasks**: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)

### Fine-Tuned Model
- **Task**: Sentiment Analysis
- **Labels**: Neutral, Positive, Negative
- **Frameworks**: Hugging Face Transformers, PEFT -Parameter-Efficient FineTuning
- **Methodology**: Low-Rank Adaptation (LoRA)

## Notebooks Overview

### 1. Fine-Tuning (`fine-tuner.ipynb`)
This notebook fine-tunes the BERT-Base-Uncased model for sentiment analysis. Key steps include:
- Loading the base model and tokenizer.
- Preparing the dataset for training.
- Configuring training hyperparameters.
- Saving the fine-tuned model checkpoints.

### 2. Evaluation (`evaluate.ipynb`)
This notebook evaluates the fine-tuned model on a validation dataset. Key steps include:
- Loading the fine-tuned model and tokenizer.
- Preparing the validation dataset.
- Computing evaluation metrics such as accuracy and F1-score.

### 3. Inference (`inference.ipynb`)
This notebook demonstrates how to use the fine-tuned model for inference. Key steps include:
- Loading the fine-tuned model and tokenizer.
- Defining a function to predict sentiment for a batch of text inputs.
- Mapping model outputs to sentiment labels (Neutral, Positive, Negative).

### 4. Model Overview (`model-bert-base-uncased.ipynb`)
This notebook provides a comprehensive summary of the BERT-Base-Uncased model, including:
- Architecture and configuration.
- Pretraining data and tasks.
- Intended uses and limitations.
- Example usage of the base model.

### 5. Vanilla Evaluation (`evaluate-vanilla.ipynb`)
This notebook evaluates the base (non-fine-tuned) BERT model on a validation dataset. It serves as a baseline for comparison with the fine-tuned model.

### 6. Vanilla Inference (`inference-vanilla.ipynb`)
This notebook demonstrates how to use the base (non-fine-tuned) BERT model for inference. It highlights the differences in performance compared to the fine-tuned model.

## How to Use

### Fine-Tuning
1. Open `fine-tuner.ipynb`.
2. Follow the steps to fine-tune the model on your dataset.
3. Save the fine-tuned model checkpoints.

### Evaluation
1. Open `evaluate.ipynb` or `evaluate-vanilla.ipynb`.
2. Load the respective model and validation dataset.
3. Run the notebook to compute evaluation metrics.

### Inference
1. Open `inference.ipynb` or `inference-vanilla.ipynb`.
2. Load the respective model and tokenizer.
3. Use the `predict_batch` function to predict sentiment for text inputs.

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Pandas
- Jupyter Notebook

## Limitations and Bias

- **Vocabulary Limitations**: Out-of-vocabulary words are split into subwords, which may affect rare term representation.
- **Bias**: Pretraining data may reflect societal biases, which can propagate to fine-tuned models.
- **Compute Requirements**: Fine-tuning requires significant computational resources.


## Refernce

- https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
- https://huggingface.co/docs/peft/en/package_reference/lora
- https://huggingface.co/google-bert/bert-base-uncased

## Acknowledgments

- Hugging Face Transformers library
- PEFT library for efficient fine-tuning
- PyTorch framework