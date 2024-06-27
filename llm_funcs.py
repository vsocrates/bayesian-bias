from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(chain, *args, **kwargs):
    # return openai.ChatCompletion.create(**kwargs)
    return chain.invoke(*args, **kwargs)


def num_tokens_from_string(string: str, encoding_name: str="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def compute_true_bayesian_update(pre_test, lr):
    if pre_test == 1:
        odds = ((pre_test)/(1-pre_test+ 10**-10)) * lr    
    else:
        odds = ((pre_test)/(1-pre_test)) * lr
    post_test = odds / (1 + odds)
    return post_test
    