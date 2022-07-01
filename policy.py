
import numpy as np

from neural_network import preprocess
from state import ObservedState


class Policy:
    def __init__(self, epsilon, num_words, q_function):
        self.epsilon = epsilon
        self.num_words = num_words
        self.q_function = q_function

    def calc_action(self, observed_state: ObservedState):
        return self.epsilon_greedy_policy(observed_state, self.q_function, self.epsilon, self.num_words)

    @staticmethod
    def epsilon_greedy_policy(state, q_sa, epsilon, num_words) -> [int]:
        if q_sa is None or np.random.rand() < epsilon:
            idx = np.random.choice(num_words)
            return idx
        else:
            all_q_sa = q_sa.predict(preprocess(state))

            best_action_idx = np.argmax(all_q_sa)
            return best_action_idx