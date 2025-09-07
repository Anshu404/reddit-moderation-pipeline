# src/pipeline.py

# --- Standard Library Imports ---
import os
import json
import google.generativeai as genai
from typing import Optional, Dict, Any
import pandas as pd
from scipy.sparse import csr_matrix

# --- Local Application Imports ---
from .config import DataConfig, FeatureConfig, TrainConfig, GeminiConfig
from .data_loader import DataLoader
from .analysis import SQLAnalyzer, CorrelationAnalyzer
from .visualization import Visualizer
from .features import FeatureEngineer
from .modeling import ModelTrainer, Interpreter
from .gemini_integration import GeminiModerator
from .reporting import Reporter
from .utils import log_section

class Pipeline:
    """
    Orchestrates the entire end-to-end machine learning pipeline, driven by
    a set of configuration objects passed during initialization.
    """
    def __init__(self, data_cfg: DataConfig, feature_cfg: FeatureConfig, train_cfg: TrainConfig, gemini_cfg: GeminiConfig):
        """Initializes the Pipeline with all the necessary configuration objects."""
        self.data_cfg = data_cfg
        self.feature_cfg = feature_cfg
        self.train_cfg = train_cfg
        self.gemini_cfg = gemini_cfg

        # --- Data and Model Artifacts (to be populated during the run) ---
        self.df: Optional[pd.DataFrame] = None
        self.work_df: Optional[pd.DataFrame] = None
        self.X: Optional[csr_matrix] = None
        self.y: Optional[pd.Series] = None
        self.train_results: Optional[Dict[str, Dict[str, Any]]] = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.model_trainer: Optional[ModelTrainer] = None

    def run(self):
        """Executes the entire pipeline from data loading to reporting."""
        
        log_section("Step 1: Data Loading")
        loader = DataLoader(self.data_cfg)
        self.df = loader.load()

        log_section("Step 2: SQL-style Exploratory Data Analysis")
        SQLAnalyzer(self.df).run()

        log_section("Step 3: Correlation Analysis")
        corr_analyzer = CorrelationAnalyzer(self.df)
        correlation_matrix = corr_analyzer.run()
        corr_analyzer.plot_heatmap(correlation_matrix)

        log_section("Step 4: Data Visualization")
        visualizer = Visualizer(self.df)
        visualizer.core_distributions()
        visualizer.word_cloud()

        log_section("Step 5: Feature Engineering")
        self.feature_engineer = FeatureEngineer(self.feature_cfg)
        self.X, self.y, self.work_df = self.feature_engineer.transform(self.df)

        log_section("Step 6: Model Training")
        self.model_trainer = ModelTrainer(self.train_cfg)
        self.train_results = self.model_trainer.train(self.X, self.y)

        # --- THIS IS THE CORRECTED SECTION ---
        log_section("Step 7: Model Interpretation")
        interpreter = Interpreter(self.feature_engineer.tfidf_vectorizer)
        rf_model = self.model_trainer.models.get("Random Forest")
        if rf_model:
            # Get the numeric feature names from the feature engineer
            numeric_names = self.feature_engineer.numeric_feature_cols
            # Pass them to the updated get_feature_importance method
            importance_df = interpreter.get_feature_importance(rf_model, numeric_feature_names=numeric_names)
            interpreter.plot_top_features(importance_df)

        log_section("Step 8: Live Gemini API Call")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ Error: GOOGLE_API_KEY environment variable not set. Skipping Gemini call.")
        else:
            genai.configure(api_key=api_key)
            print("✅ Gemini API configured successfully.")
            
            gemini_moderator = GeminiModerator(self.gemini_cfg)
            sample_post_data = self.work_df.iloc[0]
            prompt = gemini_moderator.build_moderation_prompt(sample_post_data)
            
            print("\nSending a prompt for one sample post to the Gemini API...")
            try:
                generation_config = genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=gemini_moderator.get_moderation_schema()
                )
                model = genai.GenerativeModel(
                    model_name=gemini_moderator.model_name,
                    generation_config=generation_config
                )
                response = model.generate_content(prompt)
                
                print("\n✅ Received response from Gemini:")
                moderation_decision = json.loads(response.text)
                print(json.dumps(moderation_decision, indent=2))
            except Exception as e:
                print(f"\n❌ An error occurred while calling the Gemini API: {e}")

        log_section("Step 9: Final Reporting")
        reporter = Reporter(self.work_df, self.train_results)
        reporter.report()

# # src/pipeline.py

# # --- Standard Library Imports ---
# import os
# import json
# import google.generativeai as genai
# from typing import Optional, Dict, Any
# import pandas as pd
# from scipy.sparse import csr_matrix

# # --- Local Application Imports ---
# from .config import (
#     DataConfig, FeatureConfig, TrainConfig, GeminiConfig, RANDOM_STATE
# )
# from .data_loader import DataLoader
# from .analysis import SQLAnalyzer, CorrelationAnalyzer
# from .visualization import Visualizer
# from .features import FeatureEngineer
# from .modeling import ModelTrainer, Interpreter
# from .gemini_integration import GeminiModerator
# from .reporting import Reporter
# from .utils import log_section

# class Pipeline:
#     """
#     Orchestrates the entire end-to-end machine learning pipeline, driven by
#     a set of configuration objects.
#     """
#     def __init__(self, data_path: str, sample: Optional[int] = None):
#         self.data_path = data_path
#         self.sample = sample
#         self.random_state = RANDOM_STATE

#         # --- Data and Model Artifacts ---
#         self.df: Optional[pd.DataFrame] = None
#         self.work_df: Optional[pd.DataFrame] = None
#         self.X: Optional[csr_matrix] = None
#         self.y: Optional[pd.Series] = None
#         self.train_results: Optional[Dict[str, Dict[str, Any]]] = None
#         self.feature_engineer: Optional[FeatureEngineer] = None
#         self.model_trainer: Optional[ModelTrainer] = None

#     def run(self):
#         """Executes the entire pipeline from data loading to reporting."""
        
#         log_section("Step 1: Data Loading")
#         data_cfg = DataConfig(data_path=self.data_path, sample=self.sample)
#         loader = DataLoader(data_cfg)
#         self.df = loader.load()

#         log_section("Step 2: SQL-style Exploratory Data Analysis")
#         SQLAnalyzer(self.df).run()

#         log_section("Step 3: Correlation Analysis")
#         corr_analyzer = CorrelationAnalyzer(self.df)
#         correlation_matrix = corr_analyzer.run()
#         corr_analyzer.plot_heatmap(correlation_matrix)

#         log_section("Step 4: Data Visualization")
#         visualizer = Visualizer(self.df)
#         visualizer.core_distributions()
#         visualizer.word_cloud()

#         log_section("Step 5: Feature Engineering")
#         feature_cfg = FeatureConfig(text_column="title", max_text_features=5000)
#         self.feature_engineer = FeatureEngineer(feature_cfg)
#         self.X, self.y, self.work_df = self.feature_engineer.transform(self.df)

#         log_section("Step 6: Model Training")
#         train_cfg = TrainConfig(test_size=0.2, random_state=self.random_state)
#         self.model_trainer = ModelTrainer(train_cfg)
#         self.train_results = self.model_trainer.train(self.X, self.y)

#         log_section("Step 7: Model Interpretation")
#         interpreter = Interpreter(self.feature_engineer.tfidf_vectorizer)
#         rf_model = self.model_trainer.models.get("Random Forest")
#         if rf_model:
#             num_numeric = len(self.feature_engineer.numeric_feature_cols)
#             importance_df = interpreter.rf_feature_importance(rf_model, num_numeric=num_numeric)
#             interpreter.plot_top_features(importance_df)

#         log_section("Step 8: Live Gemini API Call")
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             print("❌ Error: GOOGLE_API_KEY environment variable not set. Skipping Gemini call.")
#         else:
#             genai.configure(api_key=api_key)
#             print("✅ Gemini API configured successfully.")
            
#             gemini_cfg = GeminiConfig(model_name="gemini-1.5-flash")
#             gemini_moderator = GeminiModerator(gemini_cfg)
#             sample_post_data = self.work_df.iloc[0]
#             prompt = gemini_moderator.build_moderation_prompt(sample_post_data)
            
#             print("\nSending a prompt for one sample post to the Gemini API...")
#             try:
#                 generation_config = genai.GenerationConfig(
#                     response_mime_type="application/json",
#                     response_schema=gemini_moderator.get_moderation_schema()
#                 )
#                 model = genai.GenerativeModel(
#                     model_name=gemini_moderator.model_name,
#                     generation_config=generation_config
#                 )
#                 response = model.generate_content(prompt)
                
#                 print("\n✅ Received response from Gemini:")
#                 moderation_decision = json.loads(response.text)
#                 print(json.dumps(moderation_decision, indent=2))
#             except Exception as e:
#                 print(f"\n❌ An error occurred while calling the Gemini API: {e}")

#         log_section("Step 9: Final Reporting")
#         reporter = Reporter(self.work_df, self.train_results)
#         reporter.report()