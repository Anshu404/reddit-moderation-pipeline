import pandas as pd
import numpy as np
import sys
import os

# Corrected imports: RANDOM_STATE now comes from .config
from .config import DataConfig, RANDOM_STATE
from .utils import timeit, memory_mb

class DataLoader:
    """
    Handles loading the dataset from a CSV file, with options for sampling.
    Includes error handling for missing files.
    """
    def __init__(self, cfg: DataConfig):
        """
        Initializes the DataLoader.

        Args:
            cfg (DataConfig): A configuration object with data_path and sample size.
        """
        self.cfg = cfg

    @timeit("Load dataset")
    def load(self) -> pd.DataFrame:
        """
        Loads the dataset from the path specified in the config.

        Returns:
            pd.DataFrame: The loaded pandas DataFrame.

        Raises:
            SystemExit: If the file is not found at the specified path.
        """
        try:
            df = pd.read_csv(self.cfg.data_path , low_memory=False)
            if self.cfg.sample is not None and self.cfg.sample < len(df):
                df = df.sample(self.cfg.sample, random_state=RANDOM_STATE)
            
            missing = [c for c in self.cfg.expected_columns if c not in df.columns]
            if missing:
                print(f"Warning: missing expected columns {missing}; continuing with available columns.")
            
            print(f"Loaded shape={df.shape}, mem={memory_mb(df):.2f} MB")
            return df

        except FileNotFoundError:
            print(f"❌ Error: The data file was not found at '{self.cfg.data_path}'")
            sys.exit(1) # Exit the script if the data can't be loaded
