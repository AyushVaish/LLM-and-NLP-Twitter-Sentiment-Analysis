# Twitter Sentiment Analysis with BERT & RoBERTa

## Abstract
This project analyzes tweets about coronavirus and performs sentiment analysis using transformer‑based models (BERT and RoBERTa). We fine‑tune both models on a labeled tweet dataset to classify each tweet’s sentiment as **Positive**, **Negative**, or **Neutral**. Prior to training, all tweets are deeply cleaned—removing URLs, trailing hashtags, punctuation, and other noise—to maximize model performance.

## Problem Statement
Social media sentiment provides real‑time insight into public opinion and emotion. During the COVID‑19 pandemic, understanding how people feel about government policies, health measures, and the overall situation can inform decision makers and businesses. Traditional classifiers (e.g., Naive Bayes) struggle with nuanced language, whereas transformer‑based models can capture context and subtleties in text.

**Goals:**
- Identify positive, negative, and neutral tweets about coronavirus.
- Compare a baseline Naive Bayes classifier against fine‑tuned BERT and RoBERTa.
- Quantify performance improvements and training costs.

## Dataset
- **Source**: Labeled tweet dataset on COVID‑19 sentiment  
- **Features**:
  - `tweet_id`: Unique identifier  
  - `text`: Raw tweet content  
  - `sentiment`: Label (`Positive`, `Negative`, `Neutral`)  

## Data Cleaning & Preprocessing
1. **Lowercasing**  
2. **URL removal** (e.g., `http://…`, `https://…`)  
3. **Trailing hashtag and mention removal**  
4. **Punctuation and special character stripping**  
5. **Tokenization** (using the BERT/RoBERTa tokenizer)  

These steps reduce noise and help the transformer models focus on semantic content.

## Models & Training
- **Baseline**:  
  - Multinomial Naive Bayes with TF–IDF features  
  - Achieved ~70% accuracy and F1 score  
- **Transformer Models**:  
  - **BERT (base‑uncased)**  
  - **RoBERTa (base)**  
  - Both models were fine‑tuned for **4 epochs** on a GPU  
  - **Training time**: ~11 minutes/epoch per model  
  - Total parameters fine‑tuned: >100 million  

## Results
| Model                  | Accuracy | F1 Score |
| ---------------------- | -------- | -------- |
| Naive Bayes (baseline) | ~70%     | ~70%     |
| BERT (fine‑tuned)      | ~90%     | ~90%     |
| RoBERTa (fine‑tuned)   | ~90%     | ~90%     |

> **Insight:** Deep cleaning of tweets was critical—transformers leveraged the clean text to learn richer representations, outperforming the baseline by ~20 percentage points.

## Project Structure
