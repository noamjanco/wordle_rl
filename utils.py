from typing import List
import requests
import numpy as np

def get_all_words(word_length: int = 5) -> List[str]:
    """
    Get all words allowed in play.
    :param word_length: length of words (default is 5)
    :return: List of all possible word strings in play.
    """
    word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
    words = requests.get(word_site).content.splitlines()

    words = [word for word in words if len(word) == word_length]
    words = words[:20] # reaches ~3.5 num_trials in 30 steps
    # words = list(np.array(words)[np.random.choice(len(words),100,replace=False)])
    # words = list(np.array(words)[np.random.choice(len(words),500,replace=False)])
    return words
