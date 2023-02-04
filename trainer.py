from datetime import datetime
import tensorflow as tf
import keras
import time
from neural_network import build_q_sa_model, preprocess
import numpy as np



class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.losses = []
        self.qsa_error_losses = []
        self.word_prediction_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.qsa_error_losses.append(logs.get('qsa_error_loss_function'))
        self.word_prediction_losses.append(logs.get('word_prediction_loss_function'))

class Trainer:
    def __init__(self, words, num_iterations, model_path, replay_size, n_jobs, num_plays_in_node,data_collector, epochs=100, min_generated_samples_before_training=100):
        self.words = words
        self.num_iterations = num_iterations
        self.model_path = model_path
        self.replay_size = replay_size
        self.n_jobs = n_jobs
        self.num_plays_in_node = num_plays_in_node
        self.history = LossHistory()
        self.epochs = epochs
        self.step = 0
        self._data_collector = data_collector
        self.min_generated_samples_before_training = min_generated_samples_before_training


    def run(self):
        q_sa = build_q_sa_model(num_words=len(self.words))

        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        test_summary_writer = tf.summary.create_file_writer(log_dir)
        while self.step < self.num_iterations:
            print('saving last model')
            try:
                tf.keras.models.save_model(q_sa, self.model_path)
            except:
                print('race condition, sleep for 3 sec')
                time.sleep(3)
                continue
            print('saving last model finished')

            results = self._data_collector.get_replay_buffer()
            if len(results) < self.min_generated_samples_before_training:
                print('Collected %d samples, minimum is %d. sleeping for 10 seconds'%(len(results),self.min_generated_samples_before_training))
                time.sleep(10)
                continue

            replay_states = [el for result in results for el in result.states]
            replay_actions = [el for result in results for el in result.actions]
            replay_next_states = [el for result in results for el in result.next_states]
            replay_rewards = [el for result in results for el in result.rewards]
            replay_hidden_words_idx = [el for result in results for el in result.hidden_words_idx]
            td_targets = [el for result in results for el in result.targets]
            num_trials = [len(result.states) for result in results]

            q_sa.fit(x=preprocess(replay_states),
                     y=np.array([td_targets, replay_actions, replay_hidden_words_idx]).T,
                     epochs=self.epochs,
                     callbacks=[self.history],
                     batch_size=512,
                     verbose=0)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.history.losses[-1], step=self.step)
                tf.summary.scalar('qsa_error_loss', self.history.qsa_error_losses[-1], step=self.step)
                tf.summary.scalar('word_prediction_loss', self.history.word_prediction_losses[-1], step=self.step)
                tf.summary.scalar('num_trials', np.mean(num_trials), step=self.step)
                tf.summary.scalar('reward', np.mean(replay_rewards), step=self.step)

            self.step += 1
