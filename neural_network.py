import tensorflow as tf
from tensorflow.keras import layers,models
import keras
import numpy as np
from typing import List, Union

from state import ObservedState


def preprocess(states: Union[ObservedState, List[ObservedState]]) -> np.ndarray:
    """
    Preprocess observed states
    :param states: A list or a single observed states.
    :return: Preprocessed observed states.
    """
    if not isinstance(states, List):
        states = [states]
    processed_states = np.zeros((len(states),6,5,4))
    for i, state in enumerate(states):
        processed_states[i] = np.concatenate([np.expand_dims(state.prev_actions,axis=-1),
                                              np.expand_dims(state.grey_letters,axis=-1),
                                              np.expand_dims(state.green_letters,axis=-1),
                                              np.expand_dims(state.yellow_letters,axis=-1)],axis=-1)
    return processed_states

def loss_function(y_true,y_pred):
    """

    :param y_true: N x 2 tensor, concatenation of targets and actions
    :param y_pred: N x NUM_Words tensor
    :return:
    targets: N x 1
    targets - y_pred: N x Num_Words
    """
    targets, actions = tf.split(y_true,num_or_size_splits=2,axis=-1)
    actions_one_hot = tf.one_hot(tf.cast(tf.squeeze(actions,axis=-1),tf.int32),depth=y_pred.shape[-1])
    # loss = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.multiply(tf.math.pow(targets - y_pred, 2), actions_one_hot), axis=0),axis=0)
    loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.multiply(tf.math.pow(targets - y_pred, 2), actions_one_hot), axis=-1),axis=-1)

    return loss

def build_q_sa_model(num_words: int):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(6,5,4), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))

    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))

    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_words))
    # states = keras.layers.Input((-1,6,5,4))

    # opt = keras.optimizers.SGD(learning_rate=0.00001) #previously working
    # opt = keras.optimizers.SGD(learning_rate=0.000001) #previously working
    opt = keras.optimizers.SGD(learning_rate=0.00000001) #previously working
    # opt = keras.optimizers.SGD(learning_rate=0.0001) #slightly worse
    # opt = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=opt, loss=loss_function)
    model.summary()
    return model


