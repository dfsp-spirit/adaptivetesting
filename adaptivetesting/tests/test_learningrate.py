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
from scipy.stats import t
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

    def test_item_pool(self):
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

    def test_always_same(self):
        """Test when user always answers 'same'"""
        print("Running test 'test_always_same'")
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: ["same" for _ in range(len(df))]
        )
        final_ability = ability_levels[-1][0]
        self.assertTrue(-11 <= final_ability <= 2,
                       f"Final ability {final_ability} unsrealistic for always answering 'same'.")

    def test_answers_from_server(self):
        current_source_dir = os.path.dirname(os.path.abspath(__file__)) # dev_tools
        answers_and_items_file = os.path.join(current_source_dir, "CustomTaskTrial_filtered.csv")
        df_items = pd.read_csv(answers_and_items_file)
        # assert the columns "definition", "correct", and "clicked_object" exist in the dataframe
        expected_columns = ["definition", "correct", "clicked_object"]
        for col in expected_columns:
            self.assertIn(col, df_items.columns, f"Column '{col}' not found in answers and items file.")

        # the "definition" column contains a JSON that includes the item ID in the "id" field, it looks like this:
        # {"item": {"py/object": "adaptivetesting.models.__test_item.TestItem", "id": "103", "a": 43.07178122732465, "b": 0.0119335203113762, "c": 0.121192453225549, "d": 1.0}, "correct": "diff", "id": {"py/object": "numpy.int64", "dtype": "int64", "value": 103}, "merged_file": "static/stimuli_merged/Task_0103_merged.wav", "duration_target": {"py/object": "numpy.float64",
        # parse the ID from the JSON in the "definition" column and create a new column "ItemId" in the dataframe
        import json
        def parse_item_id(definition_json: str) -> int:
            definition_dict = json.loads(definition_json)
            item_id = int(definition_dict["item"]["id"])
            return item_id

        # Create new column "ItemId"
        df_items["ItemId"] = df_items["definition"].apply(parse_item_id)

        # Make sure parsing worked: the first row must have ItemId 103, correct "diff" and clicked_object "diff"
        self.assertEqual(df_items.loc[0, "ItemId"], 103, "First row ItemId should be 103")
        self.assertEqual(df_items.loc[0, "correct"], "diff", "First row correct answer should be 'diff'")
        self.assertEqual(df_items.loc[0, "clicked_object"], "diff", "First row clicked_object should be 'diff'")


        def answer_generator(df: pd.DataFrame) -> List[str]:
            answers = []
            for idx, row in df.iterrows():
                item_id = row['ItemId']
                user_answer = row['UserAnswer']
                answers.append(user_answer)
            return answers


