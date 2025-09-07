import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

class Reporter:
    """
    Handles the final reporting and visualization of model results.
    """
    def __init__(self, df: pd.DataFrame, results: Dict[str, Dict[str, Any]]):
        """
        Initializes the Reporter with the data and model results.

        Args:
            df (pd.DataFrame): The dataframe used for analysis (with engineered features).
            results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation
                                                 metrics for each trained model.
        """
        self.df = df
        self.results = results

    def report(self):
        """
        Generates and prints the full performance report by calling helper methods.
        """
        print("\n" + "=" * 60)
        print("                 MODEL PERFORMANCE REPORT")
        print("=" * 60 + "\n")

        self._display_summary()
        self._display_model_metrics()
        self._plot_comparison()

        print("\n" + "=" * 60)
        print("                 END OF REPORT")
        print("=" * 60 + "\n")

    def _display_summary(self):
        """Displays a high-level summary of the dataset."""
        print("--- Dataset Overview ---")
        print(f"- Total posts analyzed: {len(self.df):,}")
        print(f"- Number of columns: {self.df.shape[1]}")
        if "created_utc" in self.df.columns:
            try:
                dt = pd.to_datetime(self.df["created_utc"], unit="s", errors='coerce').dropna()
                if not dt.empty:
                    time_range = f"{dt.min().strftime('%Y-%m-%d')} to {dt.max().strftime('%Y-%m-%d')}"
                    print(f"- Timestamp range: {time_range}")
            except Exception as e:
                print(f"- Could not determine timestamp range due to error: {e}")
        print("-" * 25)

    def _display_model_metrics(self):
        """Displays detailed metrics for each trained model."""
        print("\n--- Model Evaluation Results ---")
        if not self.results:
            print("No model results to display.")
            return

        for model_name, res in self.results.items():
            print(f"\n[{model_name.upper()}]")
            print(f"  Accuracy:  {res.get('accuracy', 'N/A'):.4f}")
            print(f"  F1-Score:  {res.get('f1', 'N/A'):.4f}")
            print(f"  Precision: {res.get('precision', 'N/A'):.4f}")
            print(f"  Recall:    {res.get('recall', 'N/A'):.4f}")

            cm = res.get('cm')
            print("\n  Confusion Matrix:")
            if cm is not None and cm.shape == (2, 2):
                print(f"    {'':<10}   {'Predicted Negative':<20} {'Predicted Positive'}")
                print(f"    {'Actual Negative:':<15} {cm[0][0]:<20} {cm[0][1]}")
                print(f"    {'Actual Positive:':<15} {cm[1][0]:<20} {cm[1][1]}")
            else:
                print("    Not available or invalid shape.")

            report_str = res.get('report', 'Not available.')
            print("\n  Classification Report:")
            indented_report = "\n".join(["    " + line for line in report_str.split('\n')])
            print(indented_report)
        print("-" * 25)

    def _plot_comparison(self):
        """Generates and displays a bar chart comparing model F1-scores."""
        print("\n--- Visualizing Model Comparison ---")
        if not self.results:
            print("No model results to visualize.")
            return

        labels = list(self.results.keys())
        f1_scores = [res.get('f1', 0) for res in self.results.values()]

        plt.style.use('seaborn-v0_8-talk')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(0.5)
        bars = ax.bar(labels, f1_scores, color=colors)

        ax.set_title('Model F1-Score Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylim(0, max(f1_scores) * 1.2 if f1_scores else 1)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

        plt.tight_layout()
        print("Displaying model performance chart...")
        plt.show()
