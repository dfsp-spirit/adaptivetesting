#!/usr/bin/which python

import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, conversion

# requires R (system dependency) installed and `uv add rpy2`
# To run only this test file: uv run python -m unittest adaptivetesting.tests.test_rpy2_catr

def test_rpy2():
    # 1. Load your item bank
    # Assuming CSV structure: ids, a, b, c, d
    csv_file = "adaptivetesting/tests/CustomTaskTrial_parsed_debug.csv"
    df = pd.read_csv(csv_file)
    print(f"Read data frame with {df.shape[0]}rows and {df.shape[1]} columns from file {csv_file}")
    print(f"Data frame column names: {df.columns.tolist()}")

    item_matrix_pd = df[['a', 'b', 'c', 'd']] # drop all columns except these

    # catR expects a matrix where columns are exactly: a, b, c, d
    # We drop the 'ids' column to match the required R matrix format
    with (robjects.default_converter + pandas2ri.converter).context():
        item_bank_r = robjects.conversion.get_conversion().py2rpy(item_matrix_pd)



    # 2. Set up the R bridge
    catR = importr('catR')

    # 3. Define Simulation Parameters
    # True ability (theta) of the participant
    true_theta = 1.5

    # Start List: Use 1 initial item selected by maximum information at theta=0
    start_list = robjects.ListVector({
        'nrItems': 1,
        'theta': 0,
        'startSelect': "bOpt"
    })

    # Test List: Use Expected A Posteriori (EAP) for estimation and
    # Maximum Fisher Information (MFI) for item selection
    test_list = robjects.ListVector({
        'method': "EAP",
        'itemSelect': "MFI"
    })

    # Stop List: Fixed length of 30 items
    stop_list = robjects.ListVector({
        'rule': "length",
        'thr': 30
    })

    # 4. Run the Simulation
    # randomCAT generates the full response pattern and adaptive steps automatically
    result = catR.randomCAT(
        true_theta,
        itemBank = item_bank_r,
        start = start_list,
        test = test_list,
        stop = stop_list
    )

    #print(robjects.r['summary'](result))

    # 5. Extract and Display Results
    print("--- Simulation Results ---")
    print(f"True Theta: {true_theta}")

    # Use 'thFinal' and 'seFinal' based on your R summary output
    final_theta = result.rx2('thFinal')[0]
    final_se = result.rx2('seFinal')[0]

    print(f"Estimated Theta: {final_theta}")
    print(f"Standard Error: {final_se}")

    # Items seen (indices) - key is 'testItems'
    items_administered = list(result.rx2('testItems'))
    print(f"Items Administered (Indices): {items_administered}")

    # Responses (0/1) - key is 'pattern'
    responses = list(result.rx2('pattern'))
    print(f"Responses: {responses}")

    # Optional: If you want the estimate history (all 30 steps)
    # theta_history = list(result.rx2('thetaProv'))
if __name__ == "__main__":
    test_rpy2()