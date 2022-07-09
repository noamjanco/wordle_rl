from state import GameState

REWARD_WIN = 100
REWARD_LOSE = -100

def calc_reward(state: GameState, action: str) -> float:
    """
    Reward function, given game state and action returns a scalar reward
    :param state: GameState describing the game state
    :param action: Action str, which is the guessed word
    :return: reward scalar
    """
    if action == state.hidden_state:
        return REWARD_WIN
    else:
        if state.observed_state.trial == 5:
            return REWARD_LOSE
        total_reward = 0
        for i in range(5):
            total_reward += (action[i] == state.hidden_state[i])
            for j in range(5):
                if i == j:
                    continue
                else:
                    total_reward += 0.1 * (action[i] == state.hidden_state[j])
        return total_reward - state.observed_state.trial * 10