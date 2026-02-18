import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# 1. Load your item bank
# Assuming CSV structure: ids, a, b, c, d
df = pd.read_csv("itembank.csv")

# catR expects a matrix where columns are exactly: a, b, c, d
# We drop the 'ids' column to match the required R matrix format
item_matrix_pd = df[['a', 'b', 'c', 'd']]

# 2. Set up the R bridge
pandas2ri.activate()
catR = importr('catR')

# Convert pandas dataframe to an R matrix
# We convert to R matrix specifically because catR functions expect 'matrix' types
item_bank_r = robjects.r['as.matrix'](item_matrix_pd)

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

# 5. Extract and Display Results
print("--- Simulation Results ---")
print(f"True Theta: {true_theta}")
print(f"Estimated Theta: {result.rx2('thetaEst')[0]}")
print(f"Standard Error: {result.rx2('seEst')[0]}")

# Items seen by the participant (indices)
items_administered = list(result.rx2('testItems'))
print(f"Items Administered (Indices): {items_administered}")

# Responses (0/1)
responses = list(result.rx2('pattern'))
print(f"Responses: {responses}")