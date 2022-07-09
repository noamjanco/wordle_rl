from datetime import datetime
import tensorflow as tf
import keras
import glob
import os
import time
import pickle
from neural_network import build_q_sa_model, preprocess
import numpy as np
import multiprocessing
import traceback


class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class Trainer:
    def __init__(self, words, num_iterations, model_path, data_path, replay_size, n_jobs, num_plays_in_node, epochs=100):
        self.words = words
        self.num_iterations = num_iterations
        self.model_path = model_path
        self.data_path = data_path
        self.replay_size = replay_size
        self.n_jobs = n_jobs
        self.num_plays_in_node = num_plays_in_node
        self.history = LossHistory()
        self.epochs = epochs
        self.step = 0

    def start(self):
        self.run()
        p = multiprocessing.Process(name='trainer', target=self.run)
        p.start()


    def run(self):
        q_sa = build_q_sa_model(num_words=len(self.words))

        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        test_summary_writer = tf.summary.create_file_writer(log_dir)
        plays_per_file = self.n_jobs * self.num_plays_in_node
        replay_files = int(self.replay_size / plays_per_file)
        while self.step < self.num_iterations:
            print('saving last model')
            try:
                tf.keras.models.save_model(q_sa, self.model_path)
            except:
                print('race condition, sleep for 3 sec')
                time.sleep(3)
                continue
            print('saving last model finished')

            files = list(filter(os.path.isfile, glob.glob(self.data_path + "*")))
            if len(files) == 0:
                print('No data generated yet, sleeping for 30 seconds')
                time.sleep(30)
                continue

            files.sort(key=lambda x: os.path.getmtime(x))
            if len(files) > replay_files:
                files = files[-replay_files:]
            print(files)

            results = []
            for filename in files:
                with open(filename, 'rb') as file:
                    file_results = pickle.load(file)
                    results.extend(file_results)

            replay_states = []
            replay_actions = []
            replay_rewards = []
            replay_next_states = []
            td_targets = []
            num_trials = []
            for result in results:
                states, actions, rewards, next_states, targets, trials = result
                replay_states.extend(states)
                replay_actions.extend(actions)
                replay_rewards.extend(rewards)
                replay_next_states.extend(next_states)
                td_targets.extend(targets)
                num_trials.extend(trials)

            try:
                q_sa.fit(x=preprocess(replay_states),
                         y=np.array([td_targets, replay_actions]).T,
                         epochs=self.epochs,
                         callbacks=[self.history],
                         batch_size=512,
                         verbose=0)
            except Exception:
                print(traceback.format_exc())

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.history.losses[-1], step=self.step)
                tf.summary.scalar('num_trials', np.mean(num_trials), step=self.step)

            self.step += 1
