import os
import pandas as pd
import numpy as np

from adaptivetesting.math.estimators import BayesModal, CustomPrior, NormalPrior

class HelperTools:

    silent = True

    @staticmethod
    def load_dataframe(do_postprocess: bool = False) -> pd.DataFrame:
        current_source_dir = os.path.dirname(os.path.abspath(__file__)) # dev_tools
        item_pool_file = os.path.join(current_source_dir, 'itembank_essential.csv')
        df_items = pd.read_csv(item_pool_file)

        for col in ['ids', 'correct', 'a', 'b', 'c', 'd']:
            if col not in df_items.columns:
                raise ValueError(f"CSV item bank task file '{item_pool_file}' must contain column: {col}")

        for col in ['a', 'b', 'c', 'd']:
            df_items[col] = df_items[col].astype(float)

        if do_postprocess:
            if not HelperTools.silent:
                print(" Postprocessing item parameters after loading from file...")

            # just rescale discriminations to range 0.1 to 3.0 by first normalizing to 0-1, then scaling to 0.1-3.0
            # This is a bit ad-hoc but seems to work well enough for our data.
            df_items['a'] = (df_items['a'] - df_items['a'].min()) / (df_items['a'].max() - df_items['a'].min()) # normalize to 0-1
            df_items['a'] = df_items['a'] * (3.0 - 0.1) + 0.1

            ### The difficulty parameter (b) seems fine in our data, in range -3, 3. See plots.
            ### We leave the data as is.

            ### The guessing parameter (c) seems fine (range 0.0 to 0.8), but there are many very low values close to 0.
            ### We currently dont do anything about this but setting a minimal value.
            min_reasonable_c = 0.01
            df_items['c'] = np.where(df_items['c'] < min_reasonable_c, min_reasonable_c, df_items['c'])

        # Print summary statistics for verification
        if not HelperTools.silent:
            print(f"Loaded {len(df_items)} items from item pool in file '{item_pool_file}':")
            print(f" - Discrimination (a) stats: min={df_items['a'].min()}, max={df_items['a'].max()}, mean={df_items['a'].mean()}")
            print(f" - Difficulty (b) stats: min={df_items['b'].min()}, max={df_items['b'].max()}, mean={df_items['b'].mean()}")
            print(f" - Guessing (c) stats: min={df_items['c'].min()}, max={df_items['c'].max()}, mean={df_items['c'].mean()}")

        return df_items

    @staticmethod
    def get_estimator_args():

        #est_args =  {
        #    "prior": CustomPrior(t, 100), # Use a student t distribution with 100 degrees of freedom as prior. This is close to a normal distribution, but has heavier tails, which is more robust against outliers.
        #    "optimization_interval": (-15, 15) # Ability levels outside this range are not expected and rather unrealistic. If you observe this, your item pool is probably not well calibrated. Either change it, or adapt these values.
        #}

        est_args =  {
            "prior": NormalPrior(0.0, 1.0), # Use a normal distribution with mean 0 and standard deviation 1 as prior.
        }

        return est_args
