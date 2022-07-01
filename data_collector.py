import tensorflow as tf
import os
from neural_network import loss_function, preprocess
import time
from joblib import Parallel, delayed
import pickle

from policy import Policy
from wordle_simulator import Play
import numpy as np
import multiprocessing

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


class DataCollector:
    def __init__(self, words, num_iterations, epsilon, gamma, model_path, data_path, timeout = 100, n_jobs = 5, num_plays_in_node = 50):
        self.words = words
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.gamma = gamma
        self.model_path = model_path
        self.data_path = data_path
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.num_plays_in_node = num_plays_in_node

    def start(self):
        p = multiprocessing.Process(name='data_collector', target=self.run)
        p.start()

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
            results = Parallel(n_jobs=self.n_jobs)(delayed(self.worker_task)(self.gamma, self.epsilon, self.words, self.model_path, self.num_plays_in_node)
                                                   for _ in range(self.n_jobs))
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

    def worker_task(self, gamma, epsilon, words, model_path, num_plays_in_node):
        replay_states = []
        replay_actions = []
        replay_rewards = []
        replay_next_states = []
        td_targets = []
        num_trials = []

        timeout_cnt = 0
        timeout = 3
        while timeout_cnt < timeout:
            try:
                q_sa = tf.keras.models.load_model(model_path, custom_objects={'loss_function': loss_function})
                break
            except:
                print('failed loading model due to race, sleeping for 3 seconds')
                timeout_cnt += 1
                time.sleep(3)
        if timeout_cnt == timeout:
            return [], [], [], [], [], []

        policy = Policy(epsilon, len(words), q_sa)
        play = Play(words)
        for p in range(num_plays_in_node):
            states, actions, rewards, next_states, actions_idx = play.play(policy=policy)
            # targets = compute_td_targets(states, actions, rewards, next_states, q_sa, gamma)
            targets = compute_mc_targets(states, actions, rewards, next_states, gamma)

            replay_states.extend(states)
            replay_actions.extend(actions_idx)
            replay_rewards.extend(rewards)
            replay_next_states.extend(next_states)
            td_targets.extend(targets)
            num_trials.append(len(states))

        return replay_states, replay_actions, replay_rewards, replay_next_states, td_targets, num_trials