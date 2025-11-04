
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
from adaptivetesting.models import AdaptiveTest, ItemPool
from adaptivetesting.math.estimators import BayesModal
from adaptivetesting.math.item_selection import maximum_information_criterion
from adaptivetesting.simulation import Simulation, StoppingCriterion, ResultOutputFormat

import pandas as pd
from typing import List, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from .helpers import HelperTools


class TestSimulations(unittest.TestCase):


    def _get_adaptivetest_for_theta(self, df_items, theta: float, debug: bool = False) -> AdaptiveTest:
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
            debug=debug,
            true_ability_level=theta
        )
        return adaptive_test


    def _simulate_adaptive_test_with_theta(self, theta: float, run_checks: bool = True, num_items: int = 35, sim_idx: int = 0, num_simulations: int = 1) -> Tuple[float, float]:
        """Simulate an adaptive test with a given theta (true ability level).
        Args:
            theta (float): The true ability level of the participant.
            run_checks (bool, optional): Whether to run checks so that test is aborted as soon as an estimate deviates significantly from true ability. Defaults to True. Set to False if you just want to see the full printed output for all simulations, even if an early one fails.
            num_items (int, optional): The number of test items (questions) per participant to include in the test. Defaults to 35.
            sim_idx (int, optional): Index of the simulation run, used for logging purposes only. Defaults to 0.
        """

        df_items = HelperTools.load_dataframe()

        print(df_items.sort_values('a', ascending=False).head(10))

        # Create adaptive test instance
        adaptive_test: AdaptiveTest = self._get_adaptivetest_for_theta(df_items, theta, debug=False)

        # Run simulation
        simulation = Simulation(
            test=adaptive_test,
            test_result_output=ResultOutputFormat.CSV
        )

        print(f"*Simulation #{sim_idx + 1} of {num_simulations}: Simulating for true ability level (theta): {theta} with {num_items} items.")
        simulation.simulate(
            criterion=StoppingCriterion.LENGTH,
            value=num_items  # Stop when length of test items >= 35
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

        allowed_deviation_factor = 5.0  # allow estimates within N standard errors of the true ability level
        print(f" True ability (theta): {theta}, Estimated ability: {estimate}, Standard Error: {std_err}")

        if run_checks:
            self.assertLessEqual(abs(estimate - theta), allowed_deviation_factor * std_err,
                             f"Estimated ability {estimate} is not within {allowed_deviation_factor} std_err of true ability level {theta}.")
        else:
            self.assertTrue(True)  # just a placeholder to avoid empty test case

        return estimate, std_err


    #@unittest.skip("Skipping this test as it takes long and we have written results to CSV files already.")
    def test_simulation_with_predefined_thetas_recovers_thetas_approximately(self):
        """Test that the simulation with predefined thetas recovers the thetas approximately."""
        print("Running test 'test_simulation_with_predefined_thetas_recovers_thetas_approximately'")

        num_simulations = 50
        np.random.seed(42)
        thetas = np.random.normal(0, 1, num_simulations).tolist()

        estimates: List[Tuple[float, float]] = []

        for idx, theta in enumerate(thetas):
            #print(f" Simulating for true ability level (theta): {theta}")
            estimate = self._simulate_adaptive_test_with_theta(theta, run_checks=False, num_items=35, sim_idx=idx, num_simulations=num_simulations)
            estimates.append(estimate)

        # Compute coorrelation between true thetas and estimates
        true_thetas = np.array(thetas)
        estimated_thetas = np.array([est[0] for est in estimates])
        correlation = np.corrcoef(true_thetas, estimated_thetas)[0, 1]
        print(f"Correlation between true thetas and estimated thetas: {correlation}")
        self.assertGreater(correlation, 0.85, "Correlation between true thetas and estimated thetas should be greater than 0.85.")


    def test_plot_properties(self):
        """Show pairplot of item parameters, highlighting problematic items. Useful for visual analysis of item parameter distributions, and to see which parameters or parameter combinations cause issues."""
        import seaborn as sns

        df_items = HelperTools.load_dataframe()

        problematic_ids = ["S0603", "S067", "S0709", "S101", "S1101", "S1109", "S1201", "S120", "S1212", "S129", "S132", "S1401", "S1406", "S153", "S156"]
        # Add a 'problematic' column
        df_items['problematic'] = df_items['ids'].isin(problematic_ids)

        # Pairplot colored by problematic status
        sns.pairplot(df_items, vars=['a', 'b', 'c'], hue='problematic', # ignore 'd' at it is always 1.0 in our data, so carries no information
                    diag_kind='hist', palette={True: 'red', False: 'blue'})
        #plt.show() # requires interactive environment, not script
        output_file = 'problematic_items_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved pairplot to '{output_file}'")
        plt.close()


        problematic = df_items[df_items['problematic'] == True]
        print("All items correlation:")
        print(df_items[['a', 'b', 'c']].corr())
        print("\nProblematic items correlation:")
        print(problematic[['a', 'b', 'c']].corr())


