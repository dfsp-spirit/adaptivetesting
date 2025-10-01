import unittest
import os
from adaptivetesting.implementations import TestAssembler
from adaptivetesting.models import AdaptiveTest, ItemPool, TestItem
from adaptivetesting.math.estimators import BayesModal, CustomPrior
from adaptivetesting.math.item_selection import maximum_information_criterion
from adaptivetesting.math.estimators.__functions.__estimators import probability_y1

import pandas as pd
from scipy.stats import t
from typing import List, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class TestRealWorld(unittest.TestCase):

    # Whether to postprocess item parameters when loading the item pool for tests.
    #
    # If you switch this to False, you can see that some of the tests fail.
    # If you set this to True, the item parameters are fixed up a bit to be more reasonable and tests pass.
    do_postprocess_item_parameters_in_tests = True

    @staticmethod
    def print_and_plot_item_parameters(df: pd.DataFrame, outfile_prefix: str = ""):
        """
        Load item parameters from a CSV file and generate plots and summary
        """
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert isinstance(outfile_prefix, str), "outfile_prefix must be a string"
        required_columns = {'ids', 'a', 'b', 'c'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Set up the plotting style
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribution of IRT Parameters (' + outfile_prefix + ')', fontsize=16, fontweight='bold')

        # Plot 1: Discrimination (a) parameter
        axes[0, 0].hist(df['a'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df['a'].mean(), color='red', linestyle='--', label=f'Mean: {df["a"].mean():.2f}')
        axes[0, 0].set_xlabel('Discrimination (a)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Discrimination Parameters')
        axes[0, 0].legend()
        axes[0, 0].text(0.05, 0.95, f'N = {len(df)}\nMin: {df["a"].min():.2f}\nMax: {df["a"].max():.2f}\nStd: {df["a"].std():.2f}',
                        transform=axes[0, 0].transAxes, verticalalignment='top')

        # Plot 2: Difficulty (b) parameter
        axes[0, 1].hist(df['b'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(df['b'].mean(), color='red', linestyle='--', label=f'Mean: {df["b"].mean():.2f}')
        axes[0, 1].set_xlabel('Difficulty (b)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Difficulty Parameters')
        axes[0, 1].legend()
        axes[0, 1].text(0.05, 0.95, f'N = {len(df)}\nMin: {df["b"].min():.2f}\nMax: {df["b"].max():.2f}\nStd: {df["b"].std():.2f}',
                        transform=axes[0, 1].transAxes, verticalalignment='top')

        # Plot 3: Guessing (c) parameter
        axes[1, 0].hist(df['c'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(df['c'].mean(), color='red', linestyle='--', label=f'Mean: {df["c"].mean():.6f}')
        axes[1, 0].set_xlabel('Guessing (c)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Guessing Parameters')
        axes[1, 0].legend()
        axes[1, 0].text(0.05, 0.95, f'N = {len(df)}\nMin: {df["c"].min():.6f}\nMax: {df["c"].max():.6f}\nStd: {df["c"].std():.6f}',
                        transform=axes[1, 0].transAxes, verticalalignment='top')

        # Plot 4: Box plot of all parameters (standardized for comparison)
        parameter_data = []
        for idx, row in df.iterrows():
            parameter_data.extend([
                {'Parameter': 'a', 'Value': row['a'], 'Item': row['ids']},
                {'Parameter': 'b', 'Value': row['b'], 'Item': row['ids']},
                {'Parameter': 'c', 'Value': row['c'], 'Item': row['ids']}
            ])

        param_df = pd.DataFrame(parameter_data)
        sns.boxplot(data=param_df, x='Parameter', y='Value', ax=axes[1, 1], palette=['skyblue', 'lightgreen', 'orange'])
        axes[1, 1].set_title('Box Plot of All Parameters')

        # Remove the empty subplot if you have an odd number
        # fig.delaxes(axes[1, 1])  # Uncomment if you want only 3 plots

        plt.tight_layout()
        #plt.show() # interactive only

        # Print summary statistics
        print("=" * 50)
        print(f"SUMMARY STATISTICS for item parameters ({outfile_prefix}):")
        print("=" * 50)
        print(f"Discrimination (a) parameter:")
        print(f"  Mean: {df['a'].mean():.4f}")
        print(f"  Std:  {df['a'].std():.4f}")
        print(f"  Min:  {df['a'].min():.4f}")
        print(f"  Max:  {df['a'].max():.4f}")
        print(f"  # Negative: {(df['a'] < 0).sum()} items")
        print()

        print(f"Difficulty (b) parameter:")
        print(f"  Mean: {df['b'].mean():.4f}")
        print(f"  Std:  {df['b'].std():.4f}")
        print(f"  Min:  {df['b'].min():.4f}")
        print(f"  Max:  {df['b'].max():.4f}")
        print()

        print(f"Guessing (c) parameter:")
        print(f"  Mean: {df['c'].mean():.6f}")
        print(f"  Std:  {df['c'].std():.6f}")
        print(f"  Min:  {df['c'].min():.6f}")
        print(f"  Max:  {df['c'].max():.6f}")
        print(f"  # Near zero (< 0.001): {(df['c'] < 0.001).sum()} items")
        print(f"  # Moderate (> 0.2): {(df['c'] > 0.2).sum()} items")

        #plt.savefig(outfile_prefix + 'irt_parameters_distribution.png', dpi=300, bbox_inches='tight')
        #print(f"Plots saved as '{outfile_prefix}irt_parameters_distribution.png'")
        plt.savefig(outfile_prefix + 'irt_parameters_distribution.pdf', bbox_inches='tight')
        print(f"Plots saved as '{outfile_prefix}irt_parameters_distribution.pdf'")


    def load_dataframe(self, do_postprocess: bool = False) -> pd.DataFrame:
        current_source_dir = os.path.dirname(os.path.abspath(__file__)) # dev_tools
        item_pool_file = os.path.join(current_source_dir, 'itembank_essential.csv')
        df_items = pd.read_csv(item_pool_file)

        for col in ['ids', 'correct', 'a', 'b', 'c', 'd']:
            if col not in df_items.columns:
                raise ValueError(f"CSV item bank task file '{item_pool_file}' must contain column: {col}")

        for col in ['a', 'b', 'c', 'd']:
            df_items[col] = df_items[col].astype(float)

        if do_postprocess:
            print("Postprocessing item parameters...")

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
        print(f"Loaded {len(df_items)} items from item pool in file '{item_pool_file}':")
        print(f" - Discrimination (a) stats: min={df_items['a'].min()}, max={df_items['a'].max()}, mean={df_items['a'].mean()}")
        print(f" - Difficulty (b) stats: min={df_items['b'].min()}, max={df_items['b'].max()}, mean={df_items['b'].mean()}")
        print(f" - Guessing (c) stats: min={df_items['c'].min()}, max={df_items['c'].max()}, mean={df_items['c'].mean()}")

        return df_items

    def _run_adaptive_test_with_answers(self, answer_generator):
        """Common test execution logic that takes different answer patterns"""
        df_items = self.load_dataframe(do_postprocess=self.do_postprocess_item_parameters_in_tests)
        df_items['user_answer'] = answer_generator(df_items)

        # Create item pool from dataframe
        item_pool : ItemPool = ItemPool.load_from_dataframe(df_items)

        # Create adaptive test instance
        adaptive_test: AdaptiveTest = TestAssembler(
            item_pool=item_pool,
            simulation_id="42",
            participant_id="john_doe",
            ability_estimator=BayesModal,
            estimator_args={
                "prior": CustomPrior(t, 100), # Use a student t distribution with 100 degrees of freedom as prior. This is close to a normal distribution, but has heavier tails, which is more robust against outliers.
                "optimization_interval": (-15, 15) # Ability levels outside this range are not expected and rather unrealistic. If you observe this, your item pool is probably not well calibrated. Either change it, or adapt these values.
            },
            item_selector=maximum_information_criterion,
            simulation=False,
            debug=False
        )

        # Define get_response function to simulate user answers from the dataframe
        def get_response(item: TestItem) -> int:
            correct_answer: str = df_items.loc[df_items["ids"] == item.id, "correct"].values[0]
            assert correct_answer in ["same", "diff"], f"Unexpected correct answer: {correct_answer}, expected 'same' or 'diff'."

            user_answer : str = df_items.loc[df_items["ids"] == item.id, "user_answer"].values[0]
            assert user_answer in ["same", "diff"], f"Unexpected user answer: {user_answer}, expected 'same' or 'diff'."

            user_score : int = 1 if user_answer == correct_answer else 0
            #print(f"Item ID: {item.id}, Correct Answer: {correct_answer}, User Answer: {user_answer}. Score: {user_score}")
            return user_score

        # Set the response callback
        adaptive_test.get_response = get_response

        # Run the adaptive test for each item in the pool
        ability_levels : List[Tuple[float, float]] = []
        for idx, item in enumerate(item_pool.test_items):
            adaptive_test.run_test_once()
            current_true_ability_level, std_err_estimate = adaptive_test.estimate_ability_level()
            #print(f"After item #{idx+1} with ID {item.id}: estimated ability and standard error: {current_true_ability_level}, {std_err_estimate}")
            ability_levels.append((current_true_ability_level, std_err_estimate))

        return ability_levels

    ############################ Assertion Helpers ###########################

    def _assert_monotonic_increase(self, ability_levels):
        """Assert that ability levels increase monotonically"""
        for i in range(1, len(ability_levels)):
            self.assertGreaterEqual(ability_levels[i][0], ability_levels[i-1][0],
                                  f"Ability level did not increase at index {i}: {ability_levels[i-1][0]} -> {ability_levels[i][0]}")


    def _assert_monotonic_decrease(self, ability_levels):
        """Assert that ability levels decrease monotonically"""
        for i in range(1, len(ability_levels)):
            self.assertLessEqual(ability_levels[i][0], ability_levels[i-1][0],
                               f"Ability level did not decrease at index {i}: {ability_levels[i-1][0]} -> {ability_levels[i][0]}")


    def _assert_reasonable_final_ability_for_always_answer_same(self, ability_levels):
        """
        Assert that final ability is within reasonable bounds.
        In our item pool,
        """
        final_ability = ability_levels[-1][0]
        self.assertTrue(-11 <= final_ability <= 2,
                       f"Final ability {final_ability} unsrealistic for always answering 'same'.")


    def _assert_reasonable_final_ability_for_always_answer_diff(self, ability_levels):
        """Assert that final ability is within reasonable bounds"""
        final_ability = ability_levels[-1][0]
        self.assertTrue(-11 <= final_ability <= 2,
                       f"Final ability {final_ability} unsrealistic for always answering 'diff'.")

    ############################ Test Cases ############################

    def test_item_pool(self):
        """Test that item pool loads correctly"""
        df_items = self.load_dataframe()
        self.assertGreater(len(df_items), 0, "Item pool should not be empty")
        # Check how many answers of each type are present
        num_same = (df_items['correct'] == 'same').sum()
        num_diff = (df_items['correct'] == 'diff').sum()
        self.assertEqual(num_same + num_diff, len(df_items), "All items should have a valid correct answer of 'same' or 'diff'")
        print(f"Number of answers where 'same' is correct in item pool: {num_same}")
        print(f"Number of answers where 'diff' is correct in item pool: {num_diff}")
        self.assertEqual(num_same, 55, "There should be 55 'same' answers in the item pool")
        self.assertEqual(num_diff, 90, "There should be 90 'diff' answers in the item pool")

    def test_always_same(self):
        """Test when user always answers 'same'"""
        print("Running test_always_same")
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: ["same" for _ in range(len(df))]
        )
        # For fixed "same" answers, check that final ability is reasonable
        self._assert_reasonable_final_ability_for_always_answer_same(ability_levels)

    def test_always_diff(self):
        """Test when user always answers 'diff'"""
        print("Running test_always_diff")
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: ["diff" for _ in range(len(df))]
        )
        # For fixed "diff" answers, check that final ability is reasonable
        self._assert_reasonable_final_ability_for_always_answer_diff(ability_levels)

    def test_always_correct(self):
        """Test when user always answers correctly"""
        print("Running test_always_correct")
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: df['correct'].tolist()
        )
        # When always correct, ability should increase
        self._assert_monotonic_increase(ability_levels)

    def test_always_incorrect(self):
        """Test when user always answers incorrectly"""
        print("Running test_always_incorrect")
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: ["diff" if ans == "same" else "same" for ans in df['correct'].tolist()]
        )
        # When always incorrect, ability should decrease
        self._assert_monotonic_decrease(ability_levels)

    def test_always_answering_diff_is_better_than_always_same_for_our_pool(self):
        """Test that always answering 'diff' leads to higher ability than always 'same'.
        This is to be expected for our specific item pool, which has more items were diff is correct.
        """
        print("Running test_always_answering_diff_is_better_than_always_same_for_our_pool")
        ability_levels_same = self._run_adaptive_test_with_answers(
            lambda df: ["same" for _ in range(len(df))]
        )
        ability_levels_diff = self._run_adaptive_test_with_answers(
            lambda df: ["diff" for _ in range(len(df))]
        )
        final_ability_same = ability_levels_same[-1][0]
        final_ability_diff = ability_levels_diff[-1][0]
        self.assertGreater(final_ability_diff, final_ability_same,
                           f"Final ability for always 'diff' ({final_ability_diff}) should be greater than always 'same' ({final_ability_same})")

    ### Plots ###

    def test_plot_item_parameters(self):
        """Plot item parameters before and after postprocessing. Not a real test, just for visualization."""
        df_items_post = self.load_dataframe(do_postprocess=True)
        TestRealWorld.print_and_plot_item_parameters(df=df_items_post, outfile_prefix="postprocessed_")

        df_items_raw = self.load_dataframe(do_postprocess=False)
        TestRealWorld.print_and_plot_item_parameters(df=df_items_raw, outfile_prefix="raw_")
        self.assertTrue(True, "Plotting completed successfully")
