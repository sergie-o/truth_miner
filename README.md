# 📰 TruthMiner: Fake News Detection with NLP & Transformers

<p align="center">
  <img src="https://github.com/sergie-o/truth_miner/blob/main/truthminer.png" width="900"/>
</p>

---

>## 👀 What is TruthMiner?

>In today’s world, **misinformation travels faster than facts**.  
>TruthMiner is an AI-powered system that **detects fake news headlines** using both **classic Machine Learning** and **modern Transformer models (BERT)**.  

>Think of it as your **digital truth detector** — analyzing headlines and predicting whether they’re **real** or **fake**.  

---

## 📌 Project Overview
**TruthMiner** is a Natural Language Processing (NLP) project designed to automatically classify news headlines as either **real** or **fake**.  
The project combines both **classical machine learning models** and a **fine-tuned Transformer (DistilBERT)** to benchmark performance and highlight the trade-offs between traditional approaches and modern deep learning.

---

## 🎯 Goals
- Preprocess raw text into clean, usable features  
- Apply feature engineering with **TF-IDF** and embeddings  
- Train and compare multiple ML classifiers:  
  - Logistic Regression  
  - Naïve Bayes  
  - Random Forest  
  - Support Vector Machine (SVM)  
  - XGBoost  
- Fine-tune **DistilBERT** for sequence classification  
- Evaluate models using **Accuracy, Precision, Recall, F1-score**  
- Generate predictions for the test dataset in the original file format  

---

## 📂 Dataset
- **Training data:** `dataset/training_data.csv`  
  - Format: `label<TAB>headline`  
  - `label = 0` → Fake News  
  - `label = 1` → Real News  

- **Testing data:** `dataset/testing_data.csv`  
  - Same format, but labels are placeholders (`2`)  
  - Your model predicts and replaces them with `0` or `1`

---

## ⚙️ Workflow

### 1️⃣ Preprocessing
- Convert to lowercase  
- Remove punctuation  
- Tokenize text  
- Remove stopwords  
- Apply stemming / lemmatization  

### 2️⃣ Feature Engineering
- TF-IDF vectorization (unigrams + bigrams)  
- Embeddings for Transformer input  

### 3️⃣ Modeling
- Train classical ML models with TF-IDF features  
- Fine-tune **DistilBERT** using Hugging Face `Trainer` API  
- Apply stratified splits (80/20) for validation  

### 4️⃣ Evaluation
Metrics reported:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

### 5️⃣ Prediction
- Replace `2` labels in the test set with predicted values  
- Save results as `testing_predictions.csv` (tab-separated, no header)  

---

## 📊 Results (Validation Set)

| Model                 | Accuracy | Precision | Recall | F1   |
|------------------------|----------|-----------|--------|------|
| Logistic Regression    | 0.94     | 0.94      | 0.94   | 0.94 |
| Naïve Bayes            | 0.91     | 0.90      | 0.91   | 0.90 |
| Random Forest          | 0.89     | 0.88      | 0.89   | 0.88 |
| SVM (Linear)           | **0.95** | 0.94      | 0.95   | 0.95 |
| XGBoost                | 0.93     | 0.93      | 0.93   | 0.93 |
| DistilBERT (fine-tuned)| 0.96     | 0.96      | 0.96   | 0.96 |

> ⚡ Transformers slightly outperformed classical ML, but SVM and Logistic Regression were strong baselines.

---

## 🛠️ Tech Stack
- **Languages:** Python 3.11  
- **Libraries:**  
  - `pandas`, `numpy`, `scikit-learn`  
  - `xgboost`  
  - `torch`, `transformers`, `datasets`  
- **Environment:** macOS M3 Air, VS Code  
- **Hardware:** CPU + Apple MPS (GPU acceleration)  

---

## 🚀 Usage

### Clone the repo
```bash
git clone https://github.com/yourusername/truthminer.git
cd truthminer
