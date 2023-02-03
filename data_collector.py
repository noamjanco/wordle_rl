from threading import Thread

import tensorflow as tf
import os
from neural_network import preprocess, total_loss, qsa_error_loss_function, word_prediction_loss_function
import time
from joblib import Parallel, delayed
import pickle

from policy import Policy
from wordle_simulator import Play
import numpy as np
import multiprocessing
import ray
import traceback

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


@ray.remote(num_cpus=1)
class DataCollectorActor(object):
    def __init__(self, words, epsilon, gamma):
        self.q_sa = None
        self.words = words
        self.epsilon = epsilon
        self.gamma = gamma

    def update_model(self, model_path):
        # try:
        self.q_sa = tf.keras.models.load_model(model_path, custom_objects={'total_loss': total_loss,
                                                                      'qsa_error_loss_function': qsa_error_loss_function,
                                                                      'word_prediction_loss_function': word_prediction_loss_function})
        # except Exception as e:
        #     print(traceback.format_exc())

    def generate_samples(self, num_plays):
        policy = Policy(self.epsilon, len(self.words), self.q_sa)
        play = Play(self.words)

        replay_states = []
        replay_actions = []
        replay_rewards = []
        replay_next_states = []
        replay_hidden_words_idx = []
        td_targets = []
        num_trials = []
        for p in range(num_plays):
            states, actions, rewards, next_states, actions_idx, hidden_words_idx = play.play(policy=policy)
            # targets = compute_td_targets(states, actions, rewards, next_states, q_sa, gamma)
            targets = compute_mc_targets(states, actions, rewards, next_states, self.gamma)

            replay_states.extend(states)
            replay_actions.extend(actions_idx)
            replay_rewards.extend(rewards)
            replay_next_states.extend(next_states)
            td_targets.extend(targets)
            replay_hidden_words_idx.extend(hidden_words_idx)
            num_trials.append(len(states))

        return replay_states, replay_actions, replay_rewards, replay_next_states, td_targets, num_trials, replay_hidden_words_idx

class DataCollector(Thread):
    def __init__(self, words, num_iterations, epsilon, gamma, model_path, data_path, timeout = 100, n_jobs = 5, num_plays_in_node = 50):
        Thread.__init__(self)
        self.words = words
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.gamma = gamma
        self.model_path = model_path
        self.data_path = data_path
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.num_plays_in_node = num_plays_in_node
        self.workers = [DataCollectorActor.remote(self.words, self.epsilon, self.gamma) for _ in range(self.n_jobs)]

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
            t = time.time()
            #todo: only update model on improved condition
            _ = ray.get([worker.update_model.remote(self.model_path) for worker in self.workers])

            results = ray.get([worker.generate_samples.remote(self.num_plays_in_node) for worker in self.workers])

            # results = [self.worker_task(self.gamma, self.epsilon, self.words, , self.model_path, self.num_plays_in_node)  for p in range(1)]
            dt = time.time() - t
            print('Finished generating data sampled %d, took %.2f seconds' % (cnt, dt))
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            with open(self.data_path + '%d.pkl' % cnt, 'wb') as file:
                pickle.dump(results, file)
            print('Finished generating data sample %d' % cnt)
            cnt += 1
        pass
