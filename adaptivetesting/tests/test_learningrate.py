# Tests that investigate some issues observed when using our real-world item pools and data with the adaptive testing framework:
# learning rate of adaptivetesting is way lower than for catR on same data and answer patterns.
#
#
# To run only this test file: uv run python -m unittest adaptivetesting.tests.test_learningrate


import unittest
import os
from adaptivetesting.implementations import TestAssembler
from adaptivetesting.models import AdaptiveTest, ItemPool, TestItem
from adaptivetesting.math.estimators import BayesModal, CustomPrior, NormalPrior
from adaptivetesting.math.item_selection import maximum_information_criterion
from adaptivetesting.math.estimators.__functions.__estimators import probability_y1
from adaptivetesting.simulation import Simulation, StoppingCriterion, ResultOutputFormat, SimulationPool

import pandas as pd
from scipy.stats import t, norm
from typing import List, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .helpers import HelperTools


class TestRealWorld(unittest.TestCase):


    def _run_adaptive_test_with_answers(self, answer_generator):
        """Common test execution logic that takes different answer patterns"""
        df_items = HelperTools.load_dataframe()
        df_items['user_answer'] = answer_generator(df_items)

        # Create item pool from dataframe
        item_pool : ItemPool = ItemPool.load_from_dataframe(df_items)

        # Create adaptive test instance
        adaptive_test: AdaptiveTest = TestAssembler(
            item_pool=item_pool,
            simulation_id="42",
            participant_id="john_doe",
            ability_estimator=BayesModal,
            estimator_args=HelperTools.get_estimator_args(),
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


    ############################ Test Cases ############################

    def no_test_item_pool(self):
        """Test that item pool loads correctly"""
        df_items = HelperTools.load_dataframe()
        self.assertGreater(len(df_items), 0, "Item pool should not be empty")
        # Check how many answers of each type are present
        num_same = (df_items['correct'] == 'same').sum()
        num_diff = (df_items['correct'] == 'diff').sum()
        self.assertEqual(num_same + num_diff, len(df_items), "All items should have a valid correct answer of 'same' or 'diff'")
        print(f"Number of answers where 'same' is correct in item pool: {num_same}")
        print(f"Number of answers where 'diff' is correct in item pool: {num_diff}")
        self.assertEqual(num_same, 54, f"There should be 54 'same' answers in the item pool but there are {num_same}")
        self.assertEqual(num_diff, 90, f"There should be 90 'diff' answers in the item pool but there are {num_diff}")


    def _scale_pd_series(self, series, new_min=0.5, new_max=4.0):
        """
        Scale a series from its original range to [new_min, new_max]

        Args:
            series: pandas Series to scale
            new_min: minimum of target range
            new_max: maximum of target range

        Returns:
            scaled Series
        """
        old_min = series.min()
        old_max = series.max()

        # Handle negative values by taking absolute if needed
        # (discrimination parameters should be positive in IRT)
        series = np.abs(series)

        # Scale to [0, 1] then to [new_min, new_max]
        scaled = (series - series.min()) / (series.max() - series.min())
        scaled = new_min + scaled * (new_max - new_min)

        return scaled


    def _load_server_log_dataframe(self) -> pd.DataFrame:
        current_source_dir = os.path.dirname(os.path.abspath(__file__)) # dev_tools
        answers_and_items_file = os.path.join(current_source_dir, "CustomTaskTrial_filtered.csv")
        df_answers_and_items = pd.read_csv(answers_and_items_file)

        # assert the columns "definition", "correct", and "clicked_object" exist in the dataframe
        expected_columns = ["definition", "correct", "clicked_object"]
        for col in expected_columns:
            self.assertIn(col, df_answers_and_items.columns, f"Column '{col}' not found in answers and items file.")

        df_answers_and_items = self._add_item_columns_to_dataframe(df_answers_and_items)
        expected_columns_after = expected_columns + ["ids", "a", "b", "c", "d"]
        for col in expected_columns_after:
            self.assertIn(col, df_answers_and_items.columns, f"Column '{col}' not found in answers and items file after adding item columns.")

        # Make sure parsing worked: the first row must have id 103, correct "diff" and clicked_object "diff"
        self.assertEqual(df_answers_and_items.loc[0, "ids"], 103, "First row ItemId should be 103")
        self.assertEqual(df_answers_and_items.loc[0, "correct"], "diff", "First row correct answer should be 'diff'")
        self.assertEqual(df_answers_and_items.loc[0, "clicked_object"], "diff", "First row clicked_object should be 'diff'")

        # Also check last row: ItemId 114, correct "same" and clicked_object "same"
        self.assertEqual(df_answers_and_items.loc[len(df_answers_and_items)-1, "ids"], 114, "Last row ItemId should be 143")
        self.assertEqual(df_answers_and_items.loc[len(df_answers_and_items)-1, "correct"], "diff", "Last row correct answer should be 'diff'")
        self.assertEqual(df_answers_and_items.loc[len(df_answers_and_items)-1, "clicked_object"], "diff", "Last row clicked_object should be 'diff'")

        #remove the "definition" column as it is no longer needed
        df_answers_and_items = df_answers_and_items.drop(columns=["definition"])

        #df_answers_and_items['a'] = df_answers_and_items['a']
        df_answers_and_items['a'] = self._scale_pd_series(df_answers_and_items['a'], 0.5, 4.0) # discrimination
        df_answers_and_items['b'] = self._scale_pd_series(df_answers_and_items['b'], -3.0, 3.0) # difficulty
        df_answers_and_items['c'] = self._scale_pd_series(df_answers_and_items['c'], 0.35, 0.49)   # guessing

        # write the dataframe to a new CSV for manual inspection
        debug_output_file = os.path.join(current_source_dir, "CustomTaskTrial_parsed_debug.csv")
        df_answers_and_items.to_csv(debug_output_file, index=False)

        return df_answers_and_items

    def _add_item_columns_to_dataframe(self, df_answers_and_items):
        # the "definition" column contains a JSON that includes the item ID in the "id" field, it looks like this:
        # {"item": {"py/object": "adaptivetesting.models.__test_item.TestItem", "id": "103", "a": 43.07178122732465, "b": 0.0119335203113762, "c": 0.121192453225549, "d": 1.0}, "correct": "diff", "id": {"py/object": "numpy.int64", "dtype": "int64", "value": 103}, "merged_file": "static/stimuli_merged/Task_0103_merged.wav", "duration_target": {"py/object": "numpy.float64",
        # parse the ID from the JSON in the "definition" column and create a new column "ItemId" in the dataframe
        import json
        def parse_item_id(definition_json: str) -> int:
            definition_dict = json.loads(definition_json)
            item_id = int(definition_dict["item"]["id"])
            return item_id

        def parse_item_attributes(definition_json: str) -> Tuple[float, float, float, float]:
            definition_dict = json.loads(definition_json)
            a = float(definition_dict["item"]["a"])
            b = float(definition_dict["item"]["b"])
            c = float(definition_dict["item"]["c"])
            d = float(definition_dict["item"]["d"])
            return a, b, c, d

        # Create new column "ids"
        df_answers_and_items["ids"] = df_answers_and_items["definition"].apply(parse_item_id)
        # Create new columns for item attributes
        df_answers_and_items[["a", "b", "c", "d"]] = df_answers_and_items["definition"].apply(parse_item_attributes).apply(pd.Series)
        return df_answers_and_items


    def test_answers_from_server(self):

        df_answers_and_items : pd.DataFrame = self._load_server_log_dataframe()

        do_print_raw = False

        if do_print_raw:
            for idx, row in df_answers_and_items.iterrows():
                item_id = row['ids']
                user_answer = row['clicked_object']
                correct_answer = row['correct']
                was_correct = user_answer == correct_answer
                print(f"row # {idx}, ItemID:={item_id}, User Answer={user_answer}, correct answer={correct_answer}, was_correct={was_correct}, a={row['a']}, b={row['b']}, c={row['c']}, d={row['d']}")


        item_pool : ItemPool = ItemPool.load_from_dataframe(df_answers_and_items)

        normal_prior = NormalPrior(0.0, 1.0)
        custom_prior = CustomPrior(norm, loc=0.0, scale=1.0)

        est_args =  {
            "prior": custom_prior, # Use a normal distribution with mean 0 and standard deviation 1 as prior.
            "optimization_interval": (-4, 4),

        }

        # Create adaptive test instance
        adaptive_test: AdaptiveTest = TestAssembler(
            item_pool=item_pool,
            simulation_id="42",
            participant_id="john_doe",
            ability_estimator=BayesModal,
            estimator_args=est_args, # TODO: check other options
            item_selector=maximum_information_criterion, # TODO: check alternatives
            item_selector_args={}, # TODO: check this!
            initial_ability_level=0,
            simulation=False,
            debug=False
        )

        print(f"Running adaptive test with {len(item_pool.test_items)} items from server log data...")
        print(f"Used estimator args: {est_args}")

        # Define get_response function to simulate user answers from the dataframe
        def get_score_based_on_response(item: TestItem) -> int:
            correct_answer: str = df_answers_and_items.loc[df_answers_and_items["ids"] == item.id, "correct"].values[0]
            assert correct_answer in ["same", "diff"], f"Unexpected correct answer: {correct_answer}, expected 'same' or 'diff'."

            user_answer : str = df_answers_and_items.loc[df_answers_and_items["ids"] == item.id, "clicked_object"].values[0]
            assert user_answer in ["same", "diff"], f"Unexpected user answer: {user_answer}, expected 'same' or 'diff'."

            user_score : int = 1 if user_answer == correct_answer else 0
            print(f"Item ID: {item.id}, Correct Answer: {correct_answer}, User Answer: {user_answer}. Score: {user_score}")
            return user_score

        adaptive_test.get_response = get_score_based_on_response

        # Run the adaptive test for each item in the pool
        ability_levels : List[Tuple[float, float]] = []
        for idx, item in enumerate(item_pool.test_items):
            adaptive_test.run_test_once()
            current_true_ability_level, std_err_estimate = adaptive_test.estimate_ability_level()
            print(f"After item #{idx+1} with ID {item.id}: estimated ability and standard error: {current_true_ability_level}, {std_err_estimate}")
            ability_levels.append((current_true_ability_level, std_err_estimate))

        print(f"Final ability level: {ability_levels[-1][0]}, standard error: {ability_levels[-1][1]}")


