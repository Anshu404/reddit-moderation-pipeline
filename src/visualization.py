import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from .utils import timeit

class Visualizer:
    """
    Handles the creation of all visualizations for the project,
    including EDA charts and word clouds, with a focus on professional aesthetics.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Set a professional and clean style for all plots
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")

    @timeit("Core distribution plots generation")
    def core_distributions(self):
        """
        Generates and displays a grid of core EDA plots by calling helper methods.
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Exploratory Data Analysis of Reddit Posts', fontsize=20, fontweight='bold')

        # Call individual plotting methods for better organization
        self._plot_score_distribution(axes[0, 0])
        self._plot_score_by_status(axes[0, 1])
        self._plot_top_authors(axes[0, 2])
        self._plot_awards_vs_score(axes[1, 0])
        self._plot_title_length(axes[1, 1])
        self._plot_hourly_activity(axes[1, 2])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

    def _plot_score_distribution(self, ax):
        """Plots the distribution of post scores."""
        sns.histplot(data=self.df, x="score", bins=50, ax=ax, color="steelblue", kde=True)
        ax.set_title("Post Score Distribution", fontsize=14)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_yscale("log")

    def _plot_score_by_status(self, ax):
        """Plots a boxplot of scores for active vs. deleted posts."""
        if "removed_by" in self.df.columns:
            plot_df = self.df.copy()
            plot_df['status'] = plot_df['removed_by'].apply(lambda x: 'Deleted' if pd.notna(x) else 'Active')
            sns.boxplot(data=plot_df, x='status', y='score', ax=ax, palette=["skyblue", "salmon"])
            ax.set_title("Score by Deletion Status", fontsize=14)
            ax.set_xlabel("Post Status")
            ax.set_ylabel("Score (log scale)")
            ax.set_yscale("log")

    def _plot_top_authors(self, ax):
        """Plots a bar chart of the top 20 most active authors."""
        if "author" in self.df.columns:
            top_auth = self.df["author"].value_counts().head(20)
            sns.barplot(y=top_auth.index, x=top_auth.values, ax=ax, orient='h', palette='viridis')
            ax.set_title("Top 20 Authors by Post Count", fontsize=14)
            ax.set_xlabel("Number of Posts")
            ax.set_ylabel("Author")

    def _plot_awards_vs_score(self, ax):
        """Plots a scatter plot of awards vs. score."""
        if "total_awards_received" in self.df.columns and "score" in self.df.columns:
            sns.scatterplot(
                data=self.df.sample(min(len(self.df), 5000)), # Sample for performance
                x="score",
                y="total_awards_received",
                s=20, alpha=0.5, color="darkorange", ax=ax, edgecolor=None
            )
            ax.set_title("Awards vs. Score", fontsize=14)
            ax.set_xlabel("Score (log scale)")
            ax.set_ylabel("Total Awards (log scale)")
            ax.set_xscale("log")
            ax.set_yscale("log")

    def _plot_title_length(self, ax):
        """Plots the distribution of title lengths."""
        if "title" in self.df.columns:
            title_lengths = self.df["title"].fillna("").astype(str).str.len()
            sns.histplot(title_lengths, bins=50, color="seagreen", ax=ax, kde=True)
            ax.set_title("Title Length Distribution", fontsize=14)
            ax.set_xlabel("Characters in Title")
            ax.set_ylabel("Count")

    def _plot_hourly_activity(self, ax):
        """Plots posting activity by hour of the day."""
        if "created_utc" in self.df.columns:
            hours = pd.to_datetime(self.df["created_utc"], unit="s", errors="coerce").dt.hour
            hourly_counts = hours.value_counts().sort_index()
            ax.plot(hourly_counts.index, hourly_counts.values, marker='o', linestyle='-', color="purple")
            ax.set_title("Posting Activity by Hour", fontsize=14)
            ax.set_xlabel("Hour of Day (UTC)")
            ax.set_ylabel("Number of Posts")
            ax.set_xticks(range(0, 24, 2))

    @timeit("Word cloud generation")
    def word_cloud(self):
        """Generates and displays a more insightful word cloud from post titles."""
        titles = self.df.get("title", pd.Series(dtype=str)).fillna("").astype(str)
        text = " ".join(titles.tolist())
        
        if not text.strip():
            print("No text available to generate a word cloud.")
            return

        # Custom stop words to remove common but uninteresting terms
        custom_stopwords = {"oc", "chart", "data", "visualisation", "visualization"}
        
        wc = WordCloud(
            width=1600, 
            height=800, 
            background_color="white", 
            max_words=150, 
            colormap="magma",
            stopwords=custom_stopwords
        ).generate(text)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Most Common Words in Post Titles", fontsize=18, fontweight='bold')
        plt.show()
