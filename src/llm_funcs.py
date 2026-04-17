from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(chain, *args, **kwargs):
    return chain.invoke(*args, **kwargs)


def compute_true_bayesian_update(pre_test: float, lr: float) -> float:
    """Convert pretest probability to posttest using a likelihood ratio."""
    denom = 1 - pre_test if pre_test < 1 else 1e-10
    odds = (pre_test / denom) * lr
    return odds / (1 + odds)
