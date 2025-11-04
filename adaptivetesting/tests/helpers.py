import os
import pandas as pd
import numpy as np

from adaptivetesting.math.estimators import BayesModal, CustomPrior, NormalPrior

class HelperTools:

    silent = True

    @staticmethod
    def load_dataframe() -> pd.DataFrame:
        current_source_dir = os.path.dirname(os.path.abspath(__file__)) # dev_tools
        item_pool_file = os.path.join(current_source_dir, 'itembank_essential.csv')
        df_items = pd.read_csv(item_pool_file)

        for col in ['ids', 'correct', 'a', 'b', 'c', 'd']:
            if col not in df_items.columns:
                raise ValueError(f"CSV item bank task file '{item_pool_file}' must contain column: {col}")

        for col in ['a', 'b', 'c', 'd']:
            df_items[col] = df_items[col].astype(float)

        # Print summary statistics for verification
        if not HelperTools.silent:
            print(f"Loaded {len(df_items)} items from item pool in file '{item_pool_file}':")
            print(f" - Discrimination (a) stats: min={df_items['a'].min()}, max={df_items['a'].max()}, mean={df_items['a'].mean()}")
            print(f" - Difficulty (b) stats: min={df_items['b'].min()}, max={df_items['b'].max()}, mean={df_items['b'].mean()}")
            print(f" - Guessing (c) stats: min={df_items['c'].min()}, max={df_items['c'].max()}, mean={df_items['c'].mean()}")

        return df_items

    @staticmethod
    def get_estimator_args():
        """A central place to define estimator arguments used in tests."""


        est_args =  {
            "prior": NormalPrior(0.0, 1.0), # Use a normal distribution with mean 0 and standard deviation 1 as prior.
            "optimization_interval": (-4, 4),
        }

        return est_args
