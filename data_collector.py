import threading
from threading import Thread
from typing import List

import tensorflow as tf
import os
from neural_network import preprocess, total_loss, qsa_error_loss_function, word_prediction_loss_function
import time
import copy
from policy import Policy
from wordle_simulator import Play
import numpy as np
import ray
from dataclasses import dataclass

def compute_td_targets(states, actions, rewards, next_states, prev_q_sa, gamma):
    targets = []
    for s,a,r,s_tag in zip(states, actions, rewards, next_states):
        targets.append(r + gamma * np.max(prev_q_sa.predict(preprocess(s_tag))))
    assert len(targets) == len(states)

    return targets

def compute_mc_targets(states, actions, rewards, next_states, gamma):
    targets = []
    # for s,a,r,s_tag in zip(states, actions, rewards, next_states):
    #     targets.append(r + gamma * np.max(prev_q_sa.predict(preprocess(s_tag))))
    rewards_vec = np.array(rewards)
    for i in range(len(states)):
        targets.append(np.sum(rewards_vec[i:] * np.power(gamma, np.arange(len(states)-i))))

    if len(targets) != len(states):
        print('error')
    return targets

@dataclass
class DataCollectorSample:
    """
    Dataclass for data collector sample, consists of states,
    """
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[np.ndarray]
    next_states: List[np.ndarray]
    targets: List[np.ndarray]
    hidden_words_idx: List[np.ndarray]


@ray.remote(num_cpus=1)
class DataCollectorActor(object):
    def __init__(self, words, epsilon, gamma):
        self.q_sa = None
        self.words = words
        self.epsilon = epsilon
        self.gamma = gamma

    def update_model(self, model_path):
        self.q_sa = tf.keras.models.load_model(model_path, custom_objects={'total_loss': total_loss,
                                                                      'qsa_error_loss_function': qsa_error_loss_function,
                                                                      'word_prediction_loss_function': word_prediction_loss_function})

    def generate_samples(self, num_plays):
        policy = Policy(self.epsilon, len(self.words), self.q_sa)
        play = Play(self.words)

        samples = []
        for p in range(num_plays):
            states, actions, rewards, next_states, actions_idx, hidden_words_idx = play.play(policy=policy)
            # targets = compute_td_targets(states, actions, rewards, next_states, q_sa, self.gamma)
            targets = compute_mc_targets(states, actions, rewards, next_states, self.gamma)

            samples.append(DataCollectorSample(states, actions_idx, rewards, next_states, targets, hidden_words_idx))

        return samples

class DataCollector(Thread):
    def __init__(self, words, num_iterations, epsilon, gamma, model_path, replay_size, timeout = 100, n_jobs = 5, num_plays_in_node = 50):
        Thread.__init__(self)
        self.words = words
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.gamma = gamma
        self.model_path = os.path.join(model_path, 'epoch_0')
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.num_plays_in_node = num_plays_in_node
        self.workers = [DataCollectorActor.remote(self.words, self.epsilon, self.gamma) for _ in range(self.n_jobs)]
        self._replay_buffer = []
        self.replay_size = replay_size
        self.lock = threading.Lock()

    def run(self):
        cnt = 0
        timeout_cnt = 0
        while not os.path.exists(self.model_path):
            print('waiting for model path creation, sleeping for 10 seconds')
            time.sleep(10)
            timeout_cnt += 1
            assert timeout_cnt < self.timeout, 'Timeout waiting for model path creation'

        while cnt < self.num_iterations:
            print('Started generating data sample %d' % cnt)
            #todo: only update model on improved condition
            _ = ray.get([worker.update_model.remote(self.model_path) for worker in self.workers])

            t = time.time()
            results = ray.get([worker.generate_samples.remote(self.num_plays_in_node) for worker in self.workers])
            results = [item for sublist in results for item in sublist]
            dt = time.time() - t
            print('Finished generating data sampled %d, took %.2f seconds' % (cnt, dt))


            with self.lock:
                self._replay_buffer.extend(results)
                if len(self._replay_buffer) > self.replay_size:
                    self._replay_buffer = self._replay_buffer[-self.replay_size:]

            print('Replay buffer updated with %d samples' % len(results))
            cnt += 1
        pass

    def get_replay_buffer(self):
        with self.lock:
            return copy.copy(self._replay_buffer)