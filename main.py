import argparse
import os

# Import all the necessary configuration classes and the main Pipeline
from src.config import DataConfig, FeatureConfig, TrainConfig, GeminiConfig
from src.pipeline import Pipeline
from src.utils import log_section

def main():
    """
    Main entry point for the Reddit Moderation Classifier project.

    This script configures and runs the entire end-to-end pipeline by:
    1. Parsing command-line arguments for data path and sample size.
    2. Creating configuration objects for each stage of the pipeline.
    3. Initializing the Pipeline class with these configurations.
    4. Executing the pipeline run.
    """
    # --- 1. Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run the end-to-end Reddit Post Moderation Classifier pipeline."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the dataset CSV file (e.g., data/r_dataisbeautiful_posts.csv)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of samples to use for a quick development run."
    )
    args = parser.parse_args()

    # --- 2. Create Configuration Objects ---
    # This is the central place to define all settings for the pipeline run.
    data_cfg = DataConfig(
        data_path=args.data,
        sample=args.sample
    )
    
    # --- CORRECTED LINE ---
    # Changed 'max_features' to 'max_text_features' to match the dataclass definition.
    feature_cfg = FeatureConfig(
        text_column="title",
        max_text_features=5000
    )
    # --- END OF CORRECTION ---

    # Define model hyperparameters here for easy experimentation
    train_cfg = TrainConfig(
        test_size=0.2,
        model_params={
            "Decision Tree": {
                "max_depth": 20,
                "min_samples_split": 50,
                "min_samples_leaf": 25
            },
            "Random Forest": {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 50,
                "min_samples_leaf": 25,
                "n_jobs": -1
            }
        }
    )

    gemini_cfg = GeminiConfig(
        model_name="gemini-1.5-flash"
    )

    # --- 3. Initialize and Run the Pipeline ---
    try:
        log_section("🚀 STARTING REDDIT MODERATION PIPELINE 🚀")
        
        # The pipeline is initialized with the configuration objects
        pipeline_instance = Pipeline(
            data_cfg=data_cfg,
            feature_cfg=feature_cfg,
            train_cfg=train_cfg,
            gemini_cfg=gemini_cfg
        )
        pipeline_instance.run()
        
        log_section("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY ✅")

    except FileNotFoundError:
        print(f"❌ Error: The data file was not found at the specified path: {args.data}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during pipeline execution: {e}")
        # For debugging, you might want to print the full traceback
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    main()

