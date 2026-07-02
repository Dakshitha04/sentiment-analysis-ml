# Sentiment Analysis using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based Sentiment Analysis application that classifies user text into **Positive**, **Negative**, or **Neutral** sentiments. It uses Natural Language Processing (NLP) techniques along with a Logistic Regression model to analyze text efficiently. A simple Streamlit web application allows users to enter text and receive instant sentiment predictions.

## 🎯 Features

* Clean and preprocess raw text data
* Perform sentiment classification using Machine Learning
* TF-IDF feature extraction
* Logistic Regression classifier
* Interactive Streamlit web interface
* Graphical visualization of prediction results
* Save and load trained model using Pickle

## 📂 Project Structure

```
Sentiment-Analysis/
│
├── app.py                     # Streamlit application
├── prepare_data.py            # Dataset preprocessing
├── train_model.py             # Model training script
│
├── data/
│   ├── training.1600000.processed.noemoticon.csv
│   └── dataset.csv
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── requirements.txt
├── README.md
└── screenshots/


## 🛠 Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib
* Pickle


## 📊 Machine Learning Workflow

1. Load the Twitter sentiment dataset.
2. Remove missing and invalid values.
3. Convert sentiment labels into readable classes.
4. Add neutral sentiment samples.
5. Convert text into numerical features using TF-IDF Vectorizer.
6. Split the dataset into training and testing sets.
7. Train the Logistic Regression model.
8. Evaluate model performance using Accuracy Score.
9. Save the trained model and vectorizer.
10. Predict sentiment through the Streamlit interface.


## 📁 Dataset

Dataset Used:
Twitter Sentiment140 Dataset

The original dataset contains Twitter posts labeled as positive and negative. Additional neutral samples are included to support three-class sentiment classification.


## 🤖 Machine Learning Algorithm

**Algorithm:** Logistic Regression

Logistic Regression is a supervised machine learning algorithm widely used for text classification because it is simple, efficient, and performs well on high-dimensional text data.


## 📈 Feature Extraction

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical vectors before training the model.


## 📊 Evaluation

Performance Metric:

* Accuracy Score

The trained model is evaluated on the test dataset to measure prediction accuracy.


## 💻 User Interface

The project includes a simple Streamlit application where users can:

* Enter any text or review
* Predict its sentiment instantly
* View the predicted sentiment
* Display a simple bar chart representing the prediction


## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Sentiment-Analysis.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

### Step 1: Prepare Dataset

```bash
python prepare_data.py
```

### Step 2: Train the Model

```bash
python train_model.py
```

### Step 3: Launch the Web Application

```bash
streamlit run app.py
```

## 📌 Example

**Input**

```
I absolutely love this product. It works perfectly!
```

**Output**

```
Positive
```

---

**Input**

```
This is the worst service I have ever used.
```

**Output**

```
Negative
```

---

**Input**

```
The product is okay, nothing special.
```

**Output**

```
Neutral
```

## 📚 Future Enhancements

* Use larger and more diverse datasets
* Improve text preprocessing with stemming and lemmatization
* Implement advanced deep learning models such as LSTM or BERT
* Display prediction confidence scores
* Deploy the application on Streamlit Cloud or Render
* Support multiple languages


## 📖 Conclusion

This project demonstrates how Machine Learning and Natural Language Processing can automatically classify text sentiment. It covers the complete workflow from data preprocessing and feature extraction to model training, evaluation, and deployment through a user-friendly web application. The project serves as an excellent introduction to real-world text classification using Python and Scikit-learn.


Machine Learning | Python | Data Science Enthusiast
