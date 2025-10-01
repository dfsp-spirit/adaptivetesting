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

class TestRealWorld(unittest.TestCase):

    def load_dataframe(self) -> pd.DataFrame:
        current_source_dir = os.path.dirname(os.path.abspath(__file__)) # dev_tools
        item_pool_file = os.path.join(current_source_dir, 'itembank_essential.csv')
        df_items = pd.read_csv(item_pool_file)

        for col in ['ids', 'correct', 'a', 'b', 'c', 'd']:
            if col not in df_items.columns:
                raise ValueError(f"CSV item bank task file '{item_pool_file}' must contain column: {col}")

        df_items['a'] = df_items['a'].astype(float)
        df_items['b'] = df_items['b'].astype(float)
        df_items['c'] = df_items['c'].astype(float)
        df_items['d'] = df_items['d'].astype(float)

        return df_items

    def _run_adaptive_test_with_answers(self, answer_generator):
        """Common test execution logic that takes different answer patterns"""
        df_items = self.load_dataframe()
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
                "prior": CustomPrior(t, 100),
                "optimization_interval": (-10, 10)
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
        self.assertTrue(-5 <= final_ability <= 5,
                       f"Final ability {final_ability} should be between -5 and 5")


    def _assert_reasonable_final_ability_for_always_answer_diff(self, ability_levels):
        """Assert that final ability is within reasonable bounds"""
        final_ability = ability_levels[-1][0]
        self.assertTrue(-5 <= final_ability <= 5,
                       f"Final ability {final_ability} should be between -5 and 5")

    ############################ Test Cases ############################

    def test_item_pool(self):
        """Test that item pool loads correctly"""
        df_items = self.load_dataframe()
        self.assertGreater(len(df_items), 0, "Item pool should not be empty")
        # Check how many answers of each type are present
        num_same = (df_items['correct'] == 'same').sum()
        num_diff = (df_items['correct'] == 'diff').sum()
        self.assertEqual(num_same + num_diff, len(df_items), "All items should have a valid correct answer of 'same' or 'diff'")
        print(f"Number of 'same' answers: {num_same}")
        print(f"Number of 'diff' answers: {num_diff}")
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
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: ["diff" for _ in range(len(df))]
        )
        # For fixed "diff" answers, check that final ability is reasonable
        self._assert_reasonable_final_ability_for_always_answer_diff(ability_levels)

    def test_always_correct(self):
        """Test when user always answers correctly"""
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: df['correct'].tolist()
        )
        # When always correct, ability should increase
        self._assert_monotonic_increase(ability_levels)

    def test_always_incorrect(self):
        """Test when user always answers incorrectly"""
        ability_levels = self._run_adaptive_test_with_answers(
            lambda df: ["diff" if ans == "same" else "same" for ans in df['correct'].tolist()]
        )
        # When always incorrect, ability should decrease
        self._assert_monotonic_decrease(ability_levels)