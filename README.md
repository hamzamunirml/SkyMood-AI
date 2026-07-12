# Sentiment Analysis on Customer Reviews (Airline Tweets)

<img width="1600" height="1171" alt="image" src="https://github.com/user-attachments/assets/2198fac9-50f1-4059-8820-eaaa70ba50f7" />


A Machine Learning project that classifies customer reviews as **Positive**,
**Negative**, or **Neutral** using TF-IDF feature extraction and classical
ML classifiers.

Built for a Machine Learning Internship task.

---

## 🎯 Objective

Build a model that classifies customer reviews into sentiment categories,
compare multiple classifiers, and evaluate performance using standard
classification metrics.

---

## 📊 Dataset

**Twitter US Airline Sentiment** (14,640 tweets about six major US
airlines, each labeled `positive`, `negative`, or `neutral`).

- Source: [Kaggle — Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- File used: `data/raw/Tweets.csv`
- Class distribution: 9,178 negative / 3,099 neutral / 2,363 positive (imbalanced — typical for real-world review data, since unhappy customers tweet more often)

---

## 🛠️ Tech Stack

- **Python 3.10.11**
- **pandas, numpy** — data handling
- **scikit-learn** — TF-IDF vectorization, classification models, evaluation metrics
- **NLTK** — tokenization, stopword removal, lemmatization
- **matplotlib, seaborn** — visualization
- **joblib** — model persistence

---

## 📁 Project Structure

```
Sentiment-Analysis-Reviews/
├── data/
│   ├── raw/
│   │   └── Tweets.csv                  # original dataset
│   └── processed/
│       └── cleaned_tweets.csv          # cleaned/preprocessed dataset
├── notebooks/
│   ├── sentiment_analysis.ipynb        # main notebook (local / Jupyter)
│   └── sentiment_analysis_KAGGLE.ipynb # Kaggle-ready version (auto-detects input path)
├── src/
│   ├── __init__.py
│   ├── preprocess.py                   # text cleaning & normalization
│   ├── features.py                     # TF-IDF feature extraction
│   ├── train.py                        # model training
│   ├── evaluate.py                     # metrics & confusion matrix
│   └── predict.py                      # inference on custom text
├── models/
│   ├── best_model.pkl                  # saved best-performing classifier
│   └── tfidf_vectorizer.pkl            # saved fitted TF-IDF vectorizer
├── reports/
│   └── figures/                        # generated charts (class distribution, confusion matrices, model comparison)
├── requirements.txt
└── README.md
```
## 🖥️ Live Demo

👉https://sentimentanalysisreview-hxgn76jnk3rupvhuuh6aqr.streamlit.app/
---

## 🚀 Getting Started

### 1. Set up the environment (Python 3.10.11)
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 2. Run the notebook
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```
Run all cells top to bottom. This will:
- Load and explore `data/raw/Tweets.csv`
- Clean and preprocess the text
- Extract TF-IDF features
- Train and compare 3 models
- Save the best model to `models/`
- Test the model on custom review examples

### 3. Running on Kaggle
Use `notebooks/sentiment_analysis_KAGGLE.ipynb` instead:
1. Create a new Kaggle Notebook
2. **Add Input** → search **"Twitter US Airline Sentiment"** → Add
3. Run All — the notebook auto-detects the dataset path under `/kaggle/input/`
4. Outputs (model, vectorizer, charts) are saved to `/kaggle/working/`, downloadable from the Output panel

---

## 🧠 Approach

1. **Text Cleaning** — lowercase, remove URLs/mentions/hashtags symbols/punctuation/numbers, remove stopwords, lemmatize.
2. **Feature Extraction** — TF-IDF with unigrams + bigrams, tuned config: `max_features=15000`, `sublinear_tf=True`, `min_df=2`, `max_df=0.9`.
3. **Models Trained**:
   - Multinomial Naive Bayes
   - Logistic Regression (`C=5`, tuned via `GridSearchCV`)
   - Linear SVM
4. **Evaluation** — Accuracy, weighted Precision/Recall/F1-Score, and confusion matrices for each model.
5. **Best model selected** by highest weighted F1-Score and saved via `joblib`.

---

## 📈 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** (best) | 78.9% | 78.1% | 78.9% | 78.3% |
| Linear SVM | 78.8% | 77.9% | 78.8% | 78.2% |
| Multinomial Naive Bayes | 70.9% | 73.6% | 70.9% | 64.5% |

*(Exact numbers may vary slightly by run/environment — see the notebook output for the authoritative results.)*

**Tuning notes:** Increasing `max_features` from 5000→15000 and enabling `sublinear_tf` improved F1 by roughly 1 point over the baseline TF-IDF config. `LogisticRegression`'s `C` was tuned via `GridSearchCV` (best value: `C=5`, vs. the default `C=1`). Further gains beyond ~79% accuracy would likely require word embeddings or transformer-based models (see Possible Improvements below) — TF-IDF + linear classifiers have a natural ceiling on this kind of short, informal text.

---

## 🔮 Testing on Custom Examples

The notebook includes a section that runs the saved model on custom,
hand-written review examples (not from the dataset) to sanity-check
real-world generalization. See Section 10 in the notebook.

---

## 📌 Possible Improvements

- Handle class imbalance (class weighting or SMOTE)
- Word embeddings (Word2Vec, GloVe) or transformer-based models (BERT)
- Hyperparameter tuning via GridSearchCV
- Domain-specific stopword expansion

---

## 👤 Author

Hamza Munir
BS Artificial Intelligence, KFUEIT
