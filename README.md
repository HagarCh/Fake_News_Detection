# Fake News Detection – Deep Learning Project

This project implements a **fake news classifier** using **LSTM neural networks** and **pre-trained GloVe word embeddings**.  
The goal is to classify news articles as **real** or **fake** using NLP techniques.

---

## Dataset

The dataset is taken from Kaggle:  https://www.kaggle.com/c/fake-news/data

You must download the dataset manually and place `FakeNewsNet.csv` inside:

```
data/FakeNewsNet.csv
```

This repository **does NOT** include the dataset or embeddings due to size restrictions.

### Download GloVe

GloVe is **too large to include in the repository**, so you must download it manually:  
🔗 https://www.kaggle.com/datasets/watts2/glove6b50dtxt

Place the file:

```
glove.6B.50d.txt
```

inside:

```
data/glove.6B.50d.txt
```

---


## Step 1 — Preprocessing

The text is preprocessed using:

- Lowercasing
- Stopword removal (NLTK)
- Tokenization
- Word cloud visualization

Stopwords are removed using:

```python
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
```

---

## Model Architecture

The classifier is an LSTM-based model:

- Pretrained GloVe embedding layer
- 2 stacked LSTM layers (128 units each)
- Dropout for regularization
- Dense sigmoid output layer for binary classification

```python
X = LSTM(128, return_sequences=True)(embeddings)
X = Dropout(0.3)(X)
X = LSTM(128)(X)
X = Dense(1, activation="sigmoid")(X)
```

---

## Step 2 — Training

Train/test split:

```python
train_test_split(x_train_sw, y_train, test_size=0.2)
```

Training options:

- Train a new model  
- OR load a saved pretrained model

Example training command:

```python
model.fit(X_train_indices, Y_train,
          batch_size=32,
          epochs=15,
          validation_data=val_dataset)
```

---

## Step 3 - Evaluation

The model evaluates using:

- Accuracy
- Confusion matrix (`sklearn`)

Example:

```python
loss, acc = model.evaluate(X_test_indices, Y_test)
print("Accuracy:", acc)
```

A confusion matrix is displayed for deeper insight.

---

## Saving the Model

The trained model can be saved using:

```python
tf.keras.models.save_model(model, "trainedModel")
```

Saved models can be reloaded for future use.

---

## Running the Project

1. Download the Kaggle dataset  
2. Download the GloVe embeddings  
3. Place both files inside the `/data` directory  
4. Run the script:

```bash
python fake_news_detection.py
```

---

## Requirements

Install required packages:

```bash
pip install numpy pandas tensorflow keras seaborn matplotlib wordcloud nltk scikit-learn
```

Download NLTK stopwords:

```python
import nltk
nltk.download("stopwords")
```

---

## 🖊 Author  
**Hagar Chen**
