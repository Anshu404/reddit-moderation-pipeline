import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any, Tuple

from .config import FeatureConfig
from .utils import timeit

# Ensure NLTK data is available
try:
    stopwords.words("english")
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download("stopwords", quiet=True)

class TextPreprocessor:
    """
    Handles text cleaning and normalization tasks.
    """
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def preprocess(self, text: Any) -> str:
        if pd.isna(text):
            return ""
        s = str(text).lower()
        s = re.sub(r"[^a-z\s]", " ", s)
        tokens = [w for w in s.split() if len(w) > 2 and w not in self.stop_words]
        stemmed_tokens = [self.stemmer.stem(w) for w in tokens]
        return " ".join(stemmed_tokens)

class FeatureEngineer:
    """
    Transforms the raw DataFrame into features and a target variable for modeling,
    driven by a configuration object.
    """
    def __init__(self, cfg: FeatureConfig):
        """
        Initializes the FeatureEngineer based on a configuration object.
        """
        self.cfg = cfg
        self.text_col = cfg.text_column # <-- Updated name
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=cfg.max_text_features, # <-- Updated name
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.text_preprocessor = TextPreprocessor()
        # This attribute is what your new pipeline.py needs
        self.numeric_feature_cols = [
            "score", "total_awards_received", "title_length",
            "title_word_count", "has_awards", "score_per_award", "has_flair"
        ]

    @timeit("Feature engineering and target construction")
    def transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.Series, pd.DataFrame]:
        work_df = df.copy()

        # Impute missing values
        work_df["title"] = work_df.get("title", "").fillna("")
        work_df["author"] = work_df.get("author", "unknown").fillna("unknown")
        work_df["score"] = work_df.get("score", 0).fillna(0).astype(float)
        work_df["total_awards_received"] = work_df.get("total_awards_received", 0).fillna(0).astype(float)
        work_df["author_flair_text"] = work_df.get("author_flair_text", np.nan)

        # Construct Target Variable
        work_df["should_delete"] = (
            (work_df.get("removed_by").notna()) |
            (work_df["score"] < 0) |
            (work_df["author"] == "[deleted]")
        ).astype(int)

        # Engineer Numerical and Categorical Features
        work_df["title_length"] = work_df["title"].astype(str).str.len()
        work_df["title_word_count"] = work_df["title"].astype(str).str.split().str.len().fillna(0)
        work_df["has_awards"] = (work_df["total_awards_received"] > 0).astype(int)
        work_df["score_per_award"] = work_df["score"] / (work_df["total_awards_received"] + 1)
        work_df["has_flair"] = work_df["author_flair_text"].notna().astype(int)

        # Preprocess and Vectorize Text
        work_df["processed_title"] = work_df[self.text_col].astype(str).apply(self.text_preprocessor.preprocess)
        X_text = self.tfidf_vectorizer.fit_transform(work_df["processed_title"])
        X_numeric = work_df[self.numeric_feature_cols].fillna(0).astype(float).values

        # Combine all features
        X = hstack([X_text, X_numeric], format="csr")
        y = work_df["should_delete"].astype(int)

        print(f"Final feature matrix shape: {X.shape}")
        print(f"Target variable 'should_delete' class distribution (1 = delete): {y.mean():.3f}")
        
        return X, y, work_df

