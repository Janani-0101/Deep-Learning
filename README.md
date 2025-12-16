# Deep-Learning
Deep Learning sentiment analysis using pretrained BERT model

## 📌 Project Overview
This project implements a deep learning–based text sentiment analysis model
using a pretrained BERT transformer. The goal is to classify movie reviews
as positive or negative using transfer learning.

The project demonstrates tokenization, fine-tuning of a pretrained model,
model evaluation, and inference.

---

## 📂 Dataset
- Dataset: IMDb Movie Reviews
- Task: Binary sentiment classification (Positive / Negative)

A reduced subset of the dataset was used to enable faster training.

---

## ⚙️ Technologies Used
- Python 3.11
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy

---

## 🧠 Model Details
- Pretrained Model: `bert-base-uncased`
- Architecture: Transformer-based text classifier
- Training Strategy: Transfer learning

---

## 🔍 Data Processing
- Tokenization using BERT tokenizer
- Padding and truncation to fixed sequence length
- Conversion to PyTorch tensors

---

## 🏋️ Training & Evaluation
- Model trained for 2 epochs
- Evaluation metrics:
  - Accuracy
  - F1-score
- Training logs and evaluation results are available in the notebook

---

## 💾 Model Saving
The trained model and tokenizer were saved locally for inference and reuse.

---

## ▶️ Inference Example
The trained model can predict sentiment for new text inputs.

## 🚀 How to Run
1. Install required dependencies:
2. Open the Jupyter Notebook
3. Run all cells sequentially

---

## 📌 Conclusion
This project successfully demonstrates the use of transfer learning for
text classification using a deep learning model. The pretrained BERT model
achieved reliable sentiment predictions on unseen data.
