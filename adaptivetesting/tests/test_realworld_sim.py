
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
import random
from .helpers import HelperTools


class TestSimulations(unittest.TestCase):

    do_postprocess_item_parameters_in_tests = False

    def _get_adaptivetest_for_theta(self, df_items, theta: float) -> AdaptiveTest:
        # Create adaptive test instance
        item_pool : ItemPool = ItemPool.load_from_dataframe(df_items)

        user_id = random.randint(1, 1_000_000)

        adaptive_test: AdaptiveTest = TestAssembler(
            item_pool=item_pool,
            simulation_id=f"{user_id}",
            participant_id=f"user_{user_id}",
            ability_estimator=BayesModal,
            estimator_args=HelperTools.get_estimator_args(),
            item_selector=maximum_information_criterion,
            simulation=True,
            debug=False,
            true_ability_level=theta
        )
        return adaptive_test


    def _simulate_adaptive_test_with_theta(self, theta: float):
        df_items = HelperTools.load_dataframe(do_postprocess=self.do_postprocess_item_parameters_in_tests)

        # Create adaptive test instance
        adaptive_test: AdaptiveTest = self._get_adaptivetest_for_theta(df_items, theta)

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

        allowed_deviation_factor = 3.0  # allow estimates within 3 standard errors of the true ability level

        self.assertLessEqual(abs(estimate - theta), allowed_deviation_factor * std_err,
                             f"Estimated ability {estimate} is not within {allowed_deviation_factor} std_err of true ability level {theta}.")


    def _simulate_adaptive_test_with_theta_pool_parallel(self, thetas: List[float]):
        assert isinstance(thetas, list), "thetas should be a list of float values."
        df_items = HelperTools.load_dataframe(do_postprocess=self.do_postprocess_item_parameters_in_tests)

        # Create adaptive test instances for each theta
        adaptive_tests : List[AdaptiveTest] = [self._get_adaptivetest_for_theta(df_items, theta) for theta in thetas]

        sim_pool = SimulationPool(adaptive_tests=adaptive_tests, test_result_output=ResultOutputFormat.CSV, criterion=StoppingCriterion.SE, value=0.3)
        sim_pool.start()

        ability_estimates = []
        estimation_std_errs = []
        for test, theta in zip(sim_pool.adaptive_tests, thetas):
            assert isinstance(test, AdaptiveTest), "Each test in simulation pool should be an AdaptiveTest instance."
            assert isinstance(theta, float), "Each theta should be a float value."
            estimate, std_err = test.estimate_ability_level()
            ability_estimates.append((theta, estimate, std_err))
            estimation_std_errs.append(std_err)

        # Print the real ability levels, estimates, and standard errors
        for theta, estimate, std_err in ability_estimates:
            print(f" True ability (theta): {theta:.3f}, Estimated ability: {estimate:.3f}, Standard Error: {std_err:.3f}")

        self.assertEqual(len(ability_estimates), len(thetas), "Number of ability estimates should match number of thetas.") # Fake test, this test is just about the printed output for now.


    def test_simulation_with_predefined_thetas_recovers_thetas_approximately(self):
        """Test that the simulation with predefined thetas recovers the thetas approximately."""
        print("Running test 'test_simulation_with_predefined_thetas_recovers_thetas_approximately'")
        for theta in [0.0, 1.0, 2.0]:
            print(f" Simulating for true ability level (theta): {theta}")
            self._simulate_adaptive_test_with_theta(theta)


    @unittest.skip("Skipping this test as the SimulationPool does not seem to work as expected.")
    def test_simulation_with_large_theta_pool(self):
        """Test that the simulation with predefined thetas recovers the thetas approximately."""
        print("Running test 'test_simulation_with_predefined_thetas_recovers_thetas_approximately'")
        # draw 50 thetas from a normal distribution with mean 0 and std 1
        num_simulations = 50
        np.random.seed(42)
        thetas = np.random.normal(0, 1, num_simulations).tolist()
        self._simulate_adaptive_test_with_theta_pool_parallel(thetas=thetas)

