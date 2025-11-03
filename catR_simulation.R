# Load required libraries
library(catR)
library(readr)

# Read the item bank from CSV (no preprocessing to match your Python code)
item_bank <- read_csv("./adaptivetesting/tests/itembank_essential.csv")

# Convert to matrix format expected by catR
# catR expects columns: a, b, c, d (in that order)
items_matrix <- as.matrix(item_bank[, c("a", "b", "c", "d")])
rownames(items_matrix) <- item_bank$ids

# Define simulation parameters
n_participants <- 50
test_length <- 35  # Stop after 35 items

# Generate true theta values from normal distribution (same as Python)
set.seed(42)  # For reproducibility, matching your Python seed
true_thetas <- rnorm(n_participants, mean = 0, sd = 1)

# Initialize results storage
results <- data.frame(
  true_theta = true_thetas,
  estimated_theta = numeric(n_participants),
  sem = numeric(n_participants),
  items_administered = character(n_participants),
  stringsAsFactors = FALSE
)

# Run simulations
for (i in 1:n_participants) {
  cat("*Simulation #", i, ": Simulating for true ability level (theta):",
      round(true_thetas[i], 6), "with", test_length, "items.\n")

  # Run CAT simulation with settings matching your Python code:
  # - Bayes Modal estimator (method = "BM")
  # - Normal prior with mean 0, SD 1 (priorDist = "norm", priorPar = c(0, 1))
  # - Maximum Fisher Information item selection (startSelect = "MFI")
  # - Stop after 35 items
  sim_result <- randomCAT(trueTheta = true_thetas[i],
                         itemBank = items_matrix,
                         start = list(theta = 0, startSelect = "MFI"),
                         test = list(method = "BM", priorDist = "norm", priorPar = c(0, 1)),
                         stop = list(rule = "length", thr = test_length),
                         final = list(method = "BM", priorDist = "norm", priorPar = c(0, 1)))

  # Store results
  results$estimated_theta[i] <- sim_result$thFinal
  results$sem[i] <- sim_result$seFinal

  administered_indices <- sim_result$testItems
  administered_item_ids <- rownames(items_matrix)[administered_indices]
  print(administered_item_ids)
  results$items_administered[i] <- paste(administered_item_ids, collapse = ", ")

  cat(" True ability (theta):", round(true_thetas[i], 6),
      "Estimated ability:", round(sim_result$thFinal, 6),
      "Standard Error:", round(sim_result$seFinal, 6), "\n\n")
}

# Print summary statistics
cat("\n=== SIMULATION SUMMARY ===\n")
cat("Number of simulations:", n_participants, "\n")
cat("Mean true theta:", mean(results$true_theta), "\n")
cat("Mean estimated theta:", mean(results$estimated_theta), "\n")
cat("Mean SEM:", mean(results$sem), "\n")
cat("Correlation between true and estimated theta:",
    cor(results$true_theta, results$estimated_theta), "\n")

# Check for unreasonable estimates
unreasonable <- which(abs(results$estimated_theta) > 5)
if (length(unreasonable) > 0) {
  cat("\n=== UNREASONABLE ESTIMATES FOUND ===\n")
  print(results[unreasonable, c("true_theta", "estimated_theta", "sem")])
} else {
  cat("\nNo unreasonable estimates (> |5|) found.\n")
}

# Save results to CSV for comparison with Python
write_csv(results, "catR_simulation_results.csv")