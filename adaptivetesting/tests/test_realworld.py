
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


    def test_our_issue(self):
        df_items : pd.DataFrame = self.load_dataframe()
        # Add a row with the user answers. We hardcode them here.
        df_items['user_answer'] = ["same" for _ in range(len(df_items))]  # The User always answers "same"
        item_pool : ItemPool= ItemPool.load_from_dataframe(df_items)

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
            print(f"Item ID: {item.id}, Correct Answer: {correct_answer}, User Answer: {user_answer}. Score: {user_score}")
            return user_score

        # Set the response callback
        adaptive_test.get_response = get_response

        # Run the adaptive test for each item in the pool
        for idx, item in enumerate(item_pool.test_items):
            adaptive_test.run_test_once()
            current_true_ability_level, std_err_estimate = adaptive_test.estimate_ability_level()
            print(f"After item #{idx+1} with ID {item.id}: estimated ability and standard error: {current_true_ability_level}, {std_err_estimate}")


