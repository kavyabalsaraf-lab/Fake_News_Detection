# Fake News Detection Using NLP

A Machine Learning-based fake news detection system developed using **Python**, **Natural Language Processing (NLP)**, **TF-IDF**, and **Logistic Regression** to classify news articles as fake or real.

## Features

* Fake and real news classification
* Natural Language Processing-based text preprocessing
* Text cleaning and normalization
* TF-IDF feature extraction
* Logistic Regression machine learning model
* Model accuracy and performance evaluation
* Prediction of new news articles
* Confidence score for predictions
* Flask-based web interface

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF
* Logistic Regression
* Joblib
* Flask
* HTML
* CSS
* JavaScript

## Dataset

This project uses the **ISOT Fake and Real News Dataset**, containing fake and real news articles.

The dataset includes:

* `Fake.csv` – Fake news articles
* `True.csv` – Real news articles

Important columns include:

* Title
* Text
* Subject
* Date

## Project Structure

Fake-News-Detection/
├── dataset/
│   ├── Fake.csv
│   ├── True.csv
│   ├── news.csv
│   └── processed_news.csv
│
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── prepare_dataset.py
├── preprocess.py
├── train_model.py
├── test_model.py
├── app.py
├── requirements.txt
└── README.md


Author

Kavya Balsaraf
