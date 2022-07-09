import copy

from policy import Policy
from reward import calc_reward
from state import GameState


class Play:
    def __init__(self, all_words):
        self._all_words = all_words
        self._num_words = len(self._all_words)
        self.game_state = GameState(self._all_words)


    def play(self, policy: Policy):
        self.reset()
        states = []
        actions = []
        rewards = []
        next_states = []
        actions_idx = []
        while self.game_state.observed_state.trial < 6:
            states.append(copy.deepcopy(self.game_state.observed_state))
            action_idx = policy.calc_action(self.game_state.observed_state)
            action = self._all_words[action_idx]
            actions.append(action)
            actions_idx.append(action_idx)
            reward = calc_reward(self.game_state, action)
            rewards.append(reward)
            self.game_state.update(action_idx)
            next_states.append(copy.deepcopy(self.game_state.observed_state))

            if self.game_state.is_terminal_state():
                break
        return states, actions, rewards, next_states, actions_idx

    def reset(self):
        self.game_state = GameState(self._all_words)
