from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

# A global constant for reproducibility across the project
RANDOM_STATE = 42

@dataclass
class DataConfig:
    # ... (existing DataConfig code) ...
    data_path: str
    sample: Optional[int] = None
    expected_columns: Tuple[str, ...] = (
        "id", "title", "score", "author", "author_flair_text",
        "removed_by", "total_awards_received", "awarders",
        "created_utc", "full_link",
    )

@dataclass
class FeatureConfig:
    # ... (existing FeatureConfig code) ...
    text_column: str = "title"
    max_text_features: int = 5000

@dataclass
class TrainConfig:
    # ... (existing TrainConfig code) ...
    test_size: float = 0.2
    random_state: int = RANDOM_STATE
    cv_folds: int = 5
    class_weight: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeminiConfig:
    """Configuration for the Gemini API integration."""
    # The model to use for the moderation task. Can be easily swapped.
    model_name: str = "gemini-1.5-flash"
    # Future settings like temperature or safety settings could be added here.

