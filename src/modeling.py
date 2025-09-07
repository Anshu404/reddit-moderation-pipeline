# src/modeling.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from typing import Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score, 
    precision_score, recall_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import TrainConfig
from .utils import timeit

class ModelTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    @timeit("Train and evaluate models")
    def train(self, X: csr_matrix, y: pd.Series) -> Dict[str, Dict[str, Any]]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.cfg.test_size, random_state=self.cfg.random_state, stratify=y
        )
        print(f"Data split: {X_tr.shape[0]} training samples, {X_te.shape[0]} testing samples.")
        dt_params = self.cfg.model_params.get("Decision Tree", {})
        rf_params = self.cfg.model_params.get("Random Forest", {})
        dt = DecisionTreeClassifier(random_state=self.cfg.random_state, **dt_params)
        rf = RandomForestClassifier(random_state=self.cfg.random_state, **rf_params)

        for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
            print(f"\n[Fitting] Starting to train {name}...")
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            self.models[name] = model
            self.results[name] = self._evaluate_model(y_te, y_pred)
            print(f"[{name} Evaluation] F1={self.results[name]['f1']:.3f} | Accuracy={self.results[name]['accuracy']:.3f}")
        return self.results

    # THIS IS THE CORRECTED FUNCTION
    def _evaluate_model(self, y_true, y_pred) -> Dict[str, Any]:
        """Helper function to compute all evaluation metrics."""
        return {
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "report": classification_report(y_true, y_pred, digits=3, zero_division=0),
            "cm": confusion_matrix(y_true, y_pred),
        }

class Interpreter:
    def __init__(self, tfidf_vectorizer: TfidfVectorizer):
        self.tfidf = tfidf_vectorizer

    @timeit("Extract feature importances")
    def get_feature_importance(self, model: RandomForestClassifier, numeric_feature_names: List[str]) -> pd.DataFrame:
        if not hasattr(model, "feature_importances_"): return pd.DataFrame()
        importances = model.feature_importances_
        tfidf_features = self.tfidf.get_feature_names_out()
        all_feature_names = list(tfidf_features) + numeric_feature_names
        if len(importances) != len(all_feature_names): return pd.DataFrame()
        return pd.DataFrame({"feature": all_feature_names, "importance": importances}).sort_values("importance", ascending=False)

    @staticmethod
    def plot_top_features(importance_df: pd.DataFrame, top_k: int = 25):
        if importance_df.empty: return
        top_features = importance_df.head(top_k).iloc[::-1]
        plt.figure(figsize=(10, 8))
        plt.barh(top_features["feature"], top_features["importance"], color="salmon")
        plt.title(f"Top {top_k} Feature Importances")
        plt.tight_layout()
        plt.show()




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix
# from typing import Dict, Any, List

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     accuracy_score,
#     average_precision_score,
# )
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Import from our own project modules
# from .config import TrainConfig
# from .utils import timeit

# # ==============================================================================
# # Model Training and Evaluation
# # ==============================================================================

# class ModelTrainer:
#     """
#     Handles the training and evaluation of machine learning models based on a
#     provided training configuration.
#     """
#     def __init__(self, cfg: TrainConfig):
#         """
#         Initializes the ModelTrainer with a training configuration.

#         Args:
#             cfg (TrainConfig): A dataclass object containing settings for the
#                                training process (e.g., test size, model params).
#         """
#         self.cfg = cfg
#         self.models: Dict[str, Any] = {}
#         self.results: Dict[str, Dict[str, Any]] = {}

#     @timeit("Train and evaluate models")
#     def train(self, X: csr_matrix, y: pd.Series) -> Dict[str, Dict[str, Any]]:
#         """
#         Splits data, trains models using hyperparameters from the config, and evaluates them.

#         Args:
#             X (csr_matrix): The feature matrix in sparse format.
#             y (pd_Series): The target variable.

#         Returns:
#             Dict[str, Dict[str, Any]]: A dictionary of evaluation results for each model.
#         """
#         X_tr, X_te, y_tr, y_te = train_test_split(
#             X, y, test_size=self.cfg.test_size, random_state=self.cfg.random_state, stratify=y
#         )
#         print(f"Data split: {X_tr.shape[0]} training samples, {X_te.shape[0]} testing samples.")

#         # --- Define Models Using Hyperparameters from Config ---
#         # Get model-specific params from the config, or use an empty dict if not provided.
#         dt_params = self.cfg.model_params.get("Decision Tree", {})
#         rf_params = self.cfg.model_params.get("Random Forest", {})

#         dt = DecisionTreeClassifier(
#             random_state=self.cfg.random_state,
#             class_weight=self.cfg.class_weight,
#             **dt_params  # Unpack parameters from config
#         )
#         rf = RandomForestClassifier(
#             random_state=self.cfg.random_state,
#             n_jobs=-1,
#             class_weight=self.cfg.class_weight,
#             **rf_params  # Unpack parameters from config
#         )

#         for name, model in [("Decision Tree", dt), ("Random Forest", rf)]:
#             print(f"\n[Fitting] Starting to train {name}...")
#             model.fit(X_tr, y_tr)
            
#             y_pred = model.predict(X_te)
#             y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

#             self.models[name] = model
#             self.results[name] = self._evaluate_model(y_te, y_pred, y_proba)

#             print(f"[{name} Evaluation] F1={self.results[name]['f1']:.3f} | Accuracy={self.results[name]['accuracy']:.3f}")
        
#         return self.results

#     def _evaluate_model(self, y_true, y_pred, y_proba) -> Dict[str, Any]:
#         """Helper function to compute all evaluation metrics."""
#         return {
#             "f1": f1_score(y_true, y_pred),
#             "precision": precision_score(y_true, y_pred, zero_division=0),
#             "recall": recall_score(y_true, y_pred, zero_division=0),
#             "accuracy": accuracy_score(y_true, y_pred),
#             "aupr": average_precision_score(y_true, y_proba) if y_proba is not None else np.nan,
#             "report": classification_report(y_true, y_pred, digits=3, zero_division=0),
#             "cm": confusion_matrix(y_true, y_pred),
#             "y_test": y_true, "y_pred": y_pred, "y_proba": y_proba,
#         }

# # ==============================================================================
# # Model Interpretation
# # ==============================================================================

# class Interpreter:
#     """
#     A class for interpreting trained models, focusing on feature importance.
#     """
#     def __init__(self, tfidf_vectorizer: TfidfVectorizer):
#         self.tfidf = tfidf_vectorizer

#     @timeit("Extract feature importances")
#     def get_feature_importance(self, model: RandomForestClassifier, numeric_feature_names: List[str]) -> pd.DataFrame:
#         """
#         Extracts and ranks feature importances from a trained tree-based model.

#         Args:
#             model (RandomForestClassifier): A trained model with a `feature_importances_` attribute.
#             numeric_feature_names (List[str]): The names of the numeric features, in order.

#         Returns:
#             pd.DataFrame: A DataFrame of features and their importance scores, sorted descending.
#         """
#         if not hasattr(model, "feature_importances_"):
#             print("Warning: Model does not have 'feature_importances_' attribute.")
#             return pd.DataFrame()

#         importances = model.feature_importances_
#         tfidf_features = self.tfidf.get_feature_names_out()
#         all_feature_names = list(tfidf_features) + numeric_feature_names

#         if len(importances) != len(all_feature_names):
#             print(f"Error: Mismatch in feature count. Model has {len(importances)} importances, but there are {len(all_feature_names)} feature names.")
#             return pd.DataFrame()

#         return pd.DataFrame({
#             "feature": all_feature_names,
#             "importance": importances
#         }).sort_values("importance", ascending=False)

#     @staticmethod
#     def plot_top_features(importance_df: pd.DataFrame, top_k: int = 25, title: str = "Top Feature Importances"):
#         """
#         Generates a bar plot of the top k most important features.
#         """
#         if importance_df.empty:
#             print("Cannot plot feature importances as the dataframe is empty.")
#             return
            
#         top_features = importance_df.head(top_k).iloc[::-1]
        
#         plt.figure(figsize=(10, 8))
#         plt.barh(top_features["feature"], top_features["importance"], color="salmon")
#         plt.title(title)
#         plt.xlabel("Importance Score")
#         plt.ylabel("Feature")
#         plt.tight_layout()
#         plt.show()
