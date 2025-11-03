import numpy as np
from scipy.optimize import minimize_scalar, OptimizeResult # type: ignore
from .__estimators import likelihood
from ..__prior import Prior
from ....models.__algorithm_exception import AlgorithmException
from adaptivetesting.math.estimators.__functions.__estimators import probability_y1, probability_y0

def maximize_posterior(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    response_pattern: np.ndarray,
    prior: Prior
) -> float:
    def log_posterior(mu) -> np.ndarray:
        # Use log-likelihood to prevent underflow
        p1 = probability_y1(mu, a, b, c, d)
        p0 = probability_y0(mu, a, b, c, d)

        log_terms = (response_pattern * np.log(p1 + 1e-300)) + \
                    ((1 - response_pattern) * np.log(p0 + 1e-300))

        log_lik = np.sum(log_terms)
        log_prior = np.log(prior.pdf(mu) + 1e-300)

        return log_lik + log_prior

    # Minimize negative log-posterior to maximize posterior
    result: OptimizeResult = minimize_scalar(lambda mu: -log_posterior(mu), bounds=(-4, 4), method="bounded")

    if not result.success:
        raise AlgorithmException(f"Optimization failed: {result.message}")
    else:
        return float(result.x)

