# True-and-Fake-News-Detection-using-naive-bayes-and-Logistic-regression
Using Naive Bayes & Logistic Regression (Python + scikit-learn)

## Overview
A machine learning pipeline to classify news articles as real (true) or fake using two classic models:

Multinomial Naive Bayes

Logistic Regression

The goal is to provide a clear, modular codebase for experimentation and evaluation of both algorithms with standard preprocessing: tokenization, TF‑IDF, and model tuning.

## Features
Data preprocessing: text cleaning, stopword removal, tokenization, stemming/lemmatization

Feature extraction: CountVectorizer (Bag‑of‑Words) and TfidfVectorizer

Two classifiers:

Naive Bayes (MultinomialNB with Laplace smoothing)

Logistic Regression (scikit‑learn, L2 regularization, customizable C)

Model comparison by metrics: Accuracy, Precision, Recall, F1-score, Confusion matrix



📁 Project Structure

├── data/
│   ├── train.csv
│   ├── test.csv
│   └── (optional) validation.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   ├── nb_model.pkl
│   └── lr_model.pkl
├── requirements.txt
└── README.md
# Requirements
Tested on:

Python 3.10+

scikit-learn

pandas, numpy

nltk (stopwords, stemmer or lemmatizer)


# Data Preparation
Place your labelled dataset in data/ (columns: text, label) with labels e.g. 1 = real, 0 = fake.



## Training Models
Train both models 



## Evaluation
Evaluate models on test data:


Model           Accuracy    Precision   Recall   F1-score
Naive Bayes       0.93         0.92       0.91      0.92
Logistic Reg.     0.95         0.95       0.94      0.95

## Prediction on New Samples

