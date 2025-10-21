
# Tests that investigate some issues observed when using our real-world item pools and data with the adaptive testing framework.
#
# Our item parameters are not perfect, with rather atypical value ranges, so these tests help us to understand how the framework
# behaves in these situations, and how the behavior compares to other tools for which we have setup equivalent tests,
#  like e.g. the 'catR' package in R.
#
# To run only this test file: uv run python -m unittest adaptivetesting.tests.test_realworld_sim


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


class TestSimulations(unittest.TestCase):

    do_postprocess_item_parameters_in_tests = False

    def _simulate_adaptive_test_with_theta(self, theta: float):
        df_items = HelperTools.load_dataframe(do_postprocess=self.do_postprocess_item_parameters_in_tests)

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
            simulation=True,
            debug=False,
            true_ability_level=theta
        )

        # Run simulation
        simulation = Simulation(
            test=adaptive_test,
            test_result_output=ResultOutputFormat.CSV
        )

        simulation.simulate(
            criterion=StoppingCriterion.SE,
            value=0.3  # Stop when standard error <= 0.3
        )

        # Save results
        #simulation.save_test_results()
        estimate, std_err =  simulation.test.estimate_ability_level()
        assert isinstance(estimate, float), "Estimated ability should be a float"
        assert isinstance(std_err, float), "Estimated standard error should be a float"

        # unit test assertions
        self.assertGreater(estimate, -15, f"Estimated ability {estimate} is out of reasonable bounds.")
        self.assertLess(estimate, 15, f"Estimated ability {estimate} is out of reasonable bounds.")
        self.assertGreater(std_err, 0, f"Estimated standard error {std_err} is out of reasonable bounds.")
        self.assertLess(std_err, 10, f"Estimated standard error {std_err} is out of reasonable bounds.")
        self.assertLessEqual(abs(estimate - theta), 3 * std_err,
                             f"Estimated ability {estimate} is not within 3 std_err of true ability level {theta}.")


    def test_simulation_with_predefined_thetas_recovers_thetas_approximately(self):
        """Test that the simulation with predefined thetas recovers the thetas approximately."""
        print("Running test 'test_simulation_with_predefined_thetas_recovers_thetas_approximately'")
        # TODO: Implement this test
        for theta in [0.0, 1.0, 2.0]:
            print(f" Simulating for true ability level (theta): {theta}")
            self._simulate_adaptive_test_with_theta(theta)
