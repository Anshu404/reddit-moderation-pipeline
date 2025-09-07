import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

# Importing helper functions from our utils module
from .utils import timeit, safe_print_df

class SQLAnalyzer:
    """
    Performs SQL-style exploratory data analysis on the dataframe using an
    in-memory SQLite database.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @timeit("SQL analysis (GROUP BY / HAVING / CTE / CASE)")
    def run(self) -> Dict[str, pd.DataFrame]:
        # Connect to an in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        # Load the dataframe into a SQL table named 'posts'
        self.df.to_sql("posts", conn, index=False)
        results = {}

        # Query 1: Top authors with activity and engagement metrics
        q1 = """
        SELECT author,
               COUNT(*) AS post_count,
               AVG(score) AS avg_score,
               MAX(score) AS max_score,
               SUM(COALESCE(total_awards_received, 0)) AS total_awards
        FROM posts
        WHERE author IS NOT NULL AND author != '[deleted]'
        GROUP BY author
        HAVING COUNT(*) > 10
        ORDER BY post_count DESC
        LIMIT 25
        """
        results["top_authors"] = pd.read_sql(q1, conn)

        # Query 2: Score distribution bucketed via CASE statement
        q2 = """
        SELECT
          CASE
            WHEN score BETWEEN 0 AND 100 THEN '0-100'
            WHEN score BETWEEN 101 AND 500 THEN '101-500'
            WHEN score BETWEEN 501 AND 1000 THEN '501-1000'
            WHEN score BETWEEN 1001 AND 5000 THEN '1001-5000'
            ELSE '5000+'
          END AS score_range,
          COUNT(*) AS n,
          AVG(COALESCE(total_awards_received, 0)) AS avg_awards
        FROM posts
        GROUP BY 1 /* Group by the first column (score_range) */
        ORDER BY n DESC
        """
        results["score_buckets"] = pd.read_sql(q2, conn)

        # Query 3: Deletion pattern analysis using a Common Table Expression (CTE)
        q3 = """
        WITH base AS (
          SELECT
            CASE WHEN removed_by IS NOT NULL THEN 1 ELSE 0 END AS is_deleted,
            score,
            COALESCE(total_awards_received, 0) AS awards,
            author
          FROM posts
        )
        SELECT is_deleted,
               COUNT(*) AS n,
               AVG(score) AS avg_score,
               AVG(awards) AS avg_awards,
               COUNT(DISTINCT author) AS unique_authors
        FROM base
        GROUP BY is_deleted
        ORDER BY is_deleted DESC
        """
        results["deletion_stats"] = pd.read_sql(q3, conn)

        # Query 4: Time-based aggregation by hour if 'created_utc' is present
        if "created_utc" in self.df.columns:
            q4 = """
            SELECT
              strftime('%H', datetime(created_utc, 'unixepoch')) AS hour,
              COUNT(*) AS post_count,
              AVG(score) AS avg_score
            FROM posts
            WHERE created_utc IS NOT NULL
            GROUP BY 1 /* Group by hour */
            ORDER BY hour
            """
            results["hourly"] = pd.read_sql(q4, conn)
        
        conn.close()

        # Print a summary of the generated dataframes
        for k, v in results.items():
            print(f"[SQL] {k} -> shape={v.shape}")
            safe_print_df(v)
            
        return results

class CorrelationAnalyzer:
    """
    Computes and visualizes the Pearson correlation matrix for key numerical
    and engineered features.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a new DataFrame with features needed for correlation."""
        engineered_df = pd.DataFrame()
        engineered_df["score"] = df.get("score", 0).fillna(0)
        engineered_df["total_awards_received"] = df.get("total_awards_received", 0).fillna(0)
        engineered_df["is_deleted"] = (df.get("removed_by").notna() if "removed_by" in df.columns else False).astype(int)
        engineered_df["is_author_deleted"] = (df.get("author") == "[deleted]").astype(int) if "author" in df.columns else 0
        engineered_df["title_length"] = df.get("title", "").fillna("").astype(str).str.len()
        engineered_df["has_flair"] = (df.get("author_flair_text").notna() if "author_flair_text" in df.columns else False).astype(int)
        return engineered_df

    @timeit("Compute Pearson correlations")
    def run(self) -> pd.DataFrame:
        """
        Main method to engineer features and compute the correlation matrix.
        """
        # 1. Create features specifically for this analysis
        features_df = self._engineer_features(self.df)
        
        # 2. Compute the correlation matrix on the engineered features
        corr_matrix = features_df.corr(method="pearson")
        
        print("Correlation Matrix:")
        print(corr_matrix.round(3))
        
        return corr_matrix

    @staticmethod
    def plot_heatmap(corr: pd.DataFrame):
        """Generates and displays a heatmap of the correlation matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            square=True, 
            cbar=True
        )
        plt.title("Feature Correlation Matrix (Pearson)")
        plt.tight_layout()
        plt.show()
