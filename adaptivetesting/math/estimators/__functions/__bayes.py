import numpy as np
from scipy.optimize import minimize_scalar, OptimizeResult # type: ignore
from .__estimators import likelihood
from ..__prior import Prior
from ....models.__algorithm_exception import AlgorithmException
from .__estimators import probability_y0, probability_y1


def maximize_posterior(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    response_pattern: np.ndarray,
    prior: Prior,
    optimization_interval: tuple[float, float] = (-10, 10)
) -> float:
    """_summary_

    Args:
        a (np.ndarray): item parameter a

        b (np.ndarray): item parameter b

        c (np.ndarray): item parameter c

        d (np.ndarray): item parameter d

        response_pattern (np.ndarray): response pattern (simulated or user generated)

        prior (Prior): prior distribution

        optimization_interval (Tuple[float, float]): interval used for the optimization function

    Returns:
        float: Bayes Modal estimator for the given parameters
    """
    def log_posterior(mu) -> np.ndarray:
        p1 = probability_y1(mu, a, b, c, d)
        p0 = probability_y0(mu, a, b, c, d)

        print(f"\n[DEBUG log_posterior] mu={mu}")
        print(f"  Number of items: {len(a)}")

        # Convert to arrays if needed
        p1 = np.atleast_1d(p1)
        p0 = np.atleast_1d(p0)

        # Check for problematic probabilities
        epsilon = 1e-15
        for i in range(len(a)):
            if p1[i] < 1e-10 or p0[i] < 1e-10:
                print(f"  Item {i}: a={a[i]:.2f}, b={b[i]:.3f}, resp={response_pattern[i]}")
                print(f"    p1={p1[i]:.10e}, p0={p0[i]:.10e}")
                print(f"    log(p1)={np.log(p1[i] + epsilon):.3f}, log(p0)={np.log(p0[i] + epsilon):.3f}")


        log_likelihood = np.sum((response_pattern * np.log(p1 + epsilon)) + \
                    ((1 - response_pattern) * np.log(p0 + epsilon)))
        log_prior = np.log(prior.pdf(mu) + epsilon)


        print(f"  [DEBUG log_posterior] Total log_likelihood = {log_likelihood:.3f}")
        print(f"  [DEBUG log_posterior] log_prior = {log_prior:.3f}")
        print(f"  [DEBUG log_posterior] log_posterior = {log_likelihood + log_prior:.3f}")

        return log_likelihood + log_prior

    result: OptimizeResult = minimize_scalar(lambda mu: -log_posterior(mu),
                                             bounds=optimization_interval,
                                             method="bounded") # type: ignore

    if not result.success:
        raise AlgorithmException(f"Optimization failed: {result.message}")

    else:
        return float(result.x)
