# Toxic Chat Classification

End-to-end NLP system for detecting toxic player chat using traditional ML baselines and Transformer-based models.  
Designed for real-time moderation use cases with probability-based outputs.

---

## Problem Statement

Online multiplayer platforms face significant challenges with toxic player behavior.  
This project builds a scalable text classification system to detect toxic messages in real time while balancing recall and precision tradeoffs critical for moderation systems.

---

## Dataset

- **Jigsaw Toxic Comment Classification Dataset**
- Highly imbalanced dataset (~10% toxic comments)

The dataset is not included in this repository due to size constraints.

Dataset link:  
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

---

## Approach

### 1. Exploratory Data Analysis
- Identified severe class imbalance
- Analyzed comment length and text distribution

### 2. Baseline Model
- TF-IDF + Logistic Regression
- Initial toxic recall: **0.61**
- Improved recall using class weights and decision threshold tuning

### 3. Transformer Model
- Fine-tuned **DistilBERT** for binary toxicity classification
- Achieved:
  - **Precision:** 0.85  
  - **Recall:** 0.82  
  - **F1-score:** 0.84  

---

## API Deployment

The trained model is deployed using **FastAPI**, exposing probability-based predictions to support flexible, threshold-driven moderation decisions.

### Health Check

### Predict Toxicity

#### Request
{
  "text": "i am going to kill the time"
}

#### Response
{
  "text": "i am going to kill the time",
  "toxicity_score": 0.5434,
  "toxic": true,
  "threshold_used": 0.5
}

## Tech Stack

- Python  
- Scikit-learn  
- PyTorch  
- Hugging Face Transformers  
- FastAPI  

## Key Learnings

- Handling highly imbalanced datasets in NLP classification tasks  
- Optimizing precision–recall tradeoffs for content moderation systems  
- Fine-tuning Transformer-based models for downstream classification  
- Deploying machine learning models as real-time APIs using FastAPI  


