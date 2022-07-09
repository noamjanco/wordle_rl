from typing import List
import numpy as np


class ObservedState:
    def __init__(self):
        self.trial = 0
        self.prev_actions_idx = []
        self.prev_actions_str = []
        self.prev_actions = np.zeros((6, 5))
        self.grey_letters = np.zeros((6, 5))
        self.green_letters = np.zeros((6, 5))
        self.yellow_letters = np.zeros((6, 5))

class GameState:
    def __init__(self, all_words: List[str]):
        """
        Constructor for game state.
        :param all_words: List of all allowed words in the game.
        """
        self.hidden_word_idx = np.random.choice(len(all_words))
        self.hidden_state = all_words[self.hidden_word_idx]
        self.observed_state = ObservedState()
        self.all_words = all_words

    def update(self, action_idx: int) -> None:
        """
        Update game state given an action
        :param action_idx: action index, which corresponds to the guessed word.
        :return: None.
        """
        action = self.all_words[action_idx]
        self.observed_state.prev_actions_str.append(action)
        self.observed_state.prev_actions_idx.append(action_idx)
        # action_letters = np.array([(ord(letter) - 0.5*(ord('z')+ ord('a')))/(ord('z') - ord('a')) for letter in str(action)[2:-1]])
        # self.observed_state.prev_actions[self.observed_state.trial] = action_letters

        green_letters = np.array([action_letter == true_letter for (action_letter, true_letter) in zip(action,self.hidden_state)])
        self.observed_state.green_letters[self.observed_state.trial] = green_letters
        yellow_letters = np.array([action_letter in self.hidden_state for action_letter in action])
        yellow_letters = np.logical_and(yellow_letters,np.logical_not(green_letters))

        self.observed_state.yellow_letters[self.observed_state.trial] = yellow_letters
        self.observed_state.green_letters[self.observed_state.trial] = np.logical_not(np.logical_or(green_letters, yellow_letters))
        self.observed_state.trial += 1

    def is_terminal_state(self) -> bool:
        """
        Is this a terminal state. Terminal state if number of trials exceed maximum or hidden word as guessed correctly.
        :return:
        """
        if self.observed_state.trial > 5 or self.hidden_state in self.observed_state.prev_actions_str:
            return True
        else:
            return False