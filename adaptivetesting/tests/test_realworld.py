
# Tests that investigate some issues observed when using our real-world item pools and data with the adaptive testing framework.
#
# Our item parameters are not perfect, with rather atypical value ranges, so these tests help us to understand how the framework
# behaves in these situations, and how the behavior compares to other tools for which we have setup equivalent tests,
#  like e.g. the 'catR' package in R.
#
# To run only this test file: uv run python -m unittest adaptivetesting.tests.test_realworld


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
        self.assertEqual(num_same, 55, "There should be 55 'same' answers in the item pool")
        self.assertEqual(num_diff, 90, "There should be 90 'diff' answers in the item pool")

    def test_always_same(self):
        """Test when user always answers 'same'"""
        print("Running test 'test_always_same'")
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: ["same" for _ in range(len(df))]
        )
        final_ability = ability_levels[-1][0]
        self.assertTrue(-11 <= final_ability <= 2,
                       f"Final ability {final_ability} unsrealistic for always answering 'same'.")


    def test_always_diff(self):
        """Test when user always answers 'diff'"""
        print("Running test 'test_always_diff'")
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: ["diff" for _ in range(len(df))]
        )
        final_ability = ability_levels[-1][0]
        self.assertTrue(-11 <= final_ability <= 2, f"Final ability {final_ability} unsrealistic for always answering 'diff'.")


    def test_always_answering_diff_is_better_than_always_same_for_our_pool(self):
        """Test that always answering 'diff' leads to higher ability than always 'same'.
        This is to be expected for our specific item pool, which has more items were diff is correct.
        """
        print("Running test 'test_always_answering_diff_is_better_than_always_same_for_our_pool'")
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

    def test_always_answering_diff_is_better_than_random_for_our_pool(self):
        """Test that always answering 'diff' leads to higher ability than random answers.
        This is to be expected for our specific item pool, which has more items were diff is correct.
        """
        print("Running test 'test_always_answering_diff_is_better_than_random_for_our_pool'")
        ability_levels_random = self._run_adaptive_test_with_answers(
            lambda df: np.random.choice(['same', 'diff'], size=len(df)).tolist()
        )
        ability_levels_diff = self._run_adaptive_test_with_answers(
            lambda df: ["diff" for _ in range(len(df))]
        )
        final_ability_random = ability_levels_random[-1][0]
        final_ability_diff = ability_levels_diff[-1][0]
        self.assertGreater(final_ability_diff, final_ability_random,
                           f"Final ability for always 'diff' ({final_ability_diff}) should be greater than random answers ({final_ability_random})")

    def test_itempool_guessing_parameter_for_items_with_correct_answer_same_is_higher_than_for_items_with_correct_answer_diff(self):
        """Test that the guessing parameter for items with correct answer 'same' is higher than for items with correct answer 'diff'."""
        print("Running test 'test_itempool_guessing_parameter_for_items_with_correct_answer_same_is_higher_than_for_items_with_correct_answer_diff'")
        df_items = HelperTools.load_dataframe()
        guessing_same = df_items[df_items['correct'] == 'same']['c'].mean()
        guessing_diff = df_items[df_items['correct'] == 'diff']['c'].mean()
        self.assertGreater(guessing_same, guessing_diff,
                           f"Guessing parameter for 'same' ({guessing_same}) should be greater than for 'diff' ({guessing_diff})")



