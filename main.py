from data_collector import DataCollector
from trainer import Trainer
from utils import get_all_words

words = get_all_words()

num_iterations = 10000
data_path = 'plays/'
model_path = 'q_sa_model'
timeout = 100

replay_size = 5000
n_jobs = 5
num_plays_in_node = 50
plays_per_file = n_jobs * num_plays_in_node
replay_files = int(replay_size / plays_per_file)
gamma = 1
epsilon = 0.1
epochs = 100

if __name__ == '__main__':
    data_collector = DataCollector(words=words,
                                   num_iterations=num_iterations,
                                   epsilon=epsilon,
                                   gamma=gamma,
                                   model_path=model_path,
                                   data_path=data_path,
                                   timeout=timeout,
                                   n_jobs=n_jobs,
                                   num_plays_in_node=num_plays_in_node)

    trainer = Trainer(words=words,
                      num_iterations=num_iterations,
                      model_path=model_path,
                      data_path=data_path,
                      replay_size=replay_size,
                      n_jobs=n_jobs,
                      num_plays_in_node=num_plays_in_node,
                      epochs=epochs)

    data_collector.start()
    trainer.start()