"""
Sentiment Analysis — Streamlit Web App
Classifies customer reviews as Positive, Negative, or Neutral using a
pretrained TF-IDF + Logistic Regression model.

Run with: streamlit run streamlit_app.py
(Run from the project root, so it can find src/ and models/)
"""

import os
import sys
import re

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="wide",
)

MODEL_PATH = os.path.join("models", "best_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

SENTIMENT_COLORS = {
    "positive": "#28a745",
    "negative": "#dc3545",
    "neutral": "#6c757d",
}
SENTIMENT_EMOJI = {
    "positive": "😊",
    "negative": "😠",
    "neutral": "😐",
}


# ----------------------------------------------------------------------
# Text cleaning (kept self-contained so the app works even if src/
# preprocess.py isn't on the path, e.g. when deployed standalone)
# ----------------------------------------------------------------------
@st.cache_resource
def get_preprocessor():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer


def clean_text(text: str, min_token_len: int = 3) -> str:
    stop_words, lemmatizer = get_preprocessor()

    if not isinstance(text, str) or not text:
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) >= min_token_len
    ]
    return " ".join(tokens)


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_sentiment(text, model, vectorizer):
    """Return (label, confidence_dict_or_None)"""
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    label = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        confidence = dict(zip(model.classes_, probs))

    return label, confidence


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("💬 Sentiment Analysis on Customer Reviews")
st.caption(
    "Classifies reviews as Positive, Negative, or Neutral — powered by TF-IDF + Logistic Regression"
)

model, vectorizer = load_model()

if model is None:
    st.error(
        "⚠️ Trained model not found. Please run `notebooks/sentiment_analysis.ipynb` "
        "first — it saves `models/best_model.pkl` and `models/tfidf_vectorizer.pkl`."
    )
    st.stop()

with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses a machine learning model trained on the "
        "**Twitter US Airline Sentiment** dataset (14,640 labeled tweets) "
        "to classify review text as Positive, Negative, or Neutral."
    )
    st.write(f"**Model:** `{type(model).__name__}`")
    st.write(f"**Vocabulary size:** {len(vectorizer.get_feature_names_out()):,}")
    st.markdown("---")
    st.header("🎯 Mode")
    mode = st.radio("Choose input mode", ["Single Review", "Batch (CSV Upload)"])

# ------------------------------------------------------------------
# Mode 1: Single review
# ------------------------------------------------------------------
if mode == "Single Review":
    st.subheader("📝 Enter a Review")

    example_reviews = {
        "-- Select an example --": "",
        "Positive example": "Amazing service, the crew was so friendly and helpful throughout the flight!",
        "Negative example": "The flight was delayed for 5 hours and the staff was extremely rude.",
        "Neutral example": "The flight departed and arrived on time, nothing special to mention.",
    }
    selected_example = st.selectbox(
        "Try an example, or write your own below:", list(example_reviews.keys())
    )

    default_text = example_reviews[selected_example]
    review_text = st.text_area(
        "Review text",
        value=default_text,
        height=120,
        placeholder="Type or paste a customer review here...",
    )

    if st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary"):
        if not review_text.strip():
            st.warning("⚠️ Please enter some review text.")
        else:
            with st.spinner("Analyzing..."):
                label, confidence = predict_sentiment(review_text, model, vectorizer)

            color = SENTIMENT_COLORS.get(label, "#333333")
            emoji = SENTIMENT_EMOJI.get(label, "")

            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {color}22;
                            border: 2px solid {color}; text-align: center;">
                    <h2 style="color: {color}; margin: 0;">{emoji} {label.upper()}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence:
                st.subheader("📊 Confidence Breakdown")
                conf_df = pd.DataFrame(
                    {
                        "Sentiment": list(confidence.keys()),
                        "Confidence": list(confidence.values()),
                    }
                ).sort_values("Confidence", ascending=False)
                conf_df["Confidence"] = (conf_df["Confidence"] * 100).round(1)

                fig, ax = plt.subplots(figsize=(6, 3))
                colors = [SENTIMENT_COLORS.get(s, "#333") for s in conf_df["Sentiment"]]
                ax.barh(conf_df["Sentiment"], conf_df["Confidence"], color=colors)
                ax.set_xlabel("Confidence (%)")
                ax.set_xlim(0, 100)
                for i, v in enumerate(conf_df["Confidence"]):
                    ax.text(v + 1, i, f"{v}%", va="center")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info(
                    "ℹ️ This model type doesn't provide confidence scores "
                    "(e.g. Linear SVM) — only the predicted label is shown."
                )

# ------------------------------------------------------------------
# Mode 2: Batch CSV upload
# ------------------------------------------------------------------
else:
    st.subheader("📂 Upload a CSV of Reviews")
    st.write("The CSV must contain a column with review text.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            st.stop()

        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True)

        text_column = st.selectbox(
            "Select the column containing review text:", df.columns
        )

        if st.button(
            "🔍 Analyze All Reviews", use_container_width=True, type="primary"
        ):
            with st.spinner(f"Analyzing {len(df)} reviews..."):
                texts = df[text_column].fillna("").astype(str).tolist()
                cleaned_texts = [clean_text(t) for t in texts]
                features = vectorizer.transform(cleaned_texts)
                predictions = model.predict(features)
                df["predicted_sentiment"] = predictions

            st.success(f"✅ Analyzed {len(df)} reviews!")

            col1, col2, col3 = st.columns(3)
            counts = df["predicted_sentiment"].value_counts()
            with col1:
                st.metric("😊 Positive", counts.get("positive", 0))
            with col2:
                st.metric("😠 Negative", counts.get("negative", 0))
            with col3:
                st.metric("😐 Neutral", counts.get("neutral", 0))

            fig, ax = plt.subplots(figsize=(6, 4))
            order = [
                s for s in ["negative", "neutral", "positive"] if s in counts.index
            ]
            colors = [SENTIMENT_COLORS[s] for s in order]
            ax.bar(order, [counts[s] for s in order], color=colors)
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("📋 Results")
            st.dataframe(df, use_container_width=True, height=400)

            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                data=csv,
                file_name="sentiment_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.markdown("---")
st.caption("Built with Streamlit · scikit-learn · NLTK")
