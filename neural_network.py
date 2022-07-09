import tensorflow as tf
import numpy as np
from typing import List, Union
import keras_nlp
import string
from state import ObservedState

alphabet = string.ascii_lowercase
num_features = len(alphabet) + 3

def string_vectorizer(strng):
    vector = [[0 if char != letter else 1 for char in alphabet]
                  for letter in strng]
    return vector

def preprocess(states: Union[ObservedState, List[ObservedState]]) -> np.ndarray:
    """
    Preprocess observed states
    :param states: A list or a single observed states.
    :return: Preprocessed observed states.
    """
    if not isinstance(states, List):
        states = [states]
    processed_states = np.zeros((len(states),6,5,num_features))
    for i, state in enumerate(states):
        one_hot_letter_encoding = np.zeros((6,5,len(alphabet)))
        if len(state.prev_actions_str) > 0:
            one_hot_letter_encoding[:len(state.prev_actions_str)] = np.asarray([string_vectorizer(str(action)[2:-1])
                                                                                for action in state.prev_actions_str])

        processed_states[i] = np.concatenate([one_hot_letter_encoding,
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
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((6,5*num_features),input_shape=(6,5,num_features)))
    # model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.LayerNormalization())
    model.add(keras_nlp.layers.TransformerEncoder(intermediate_dim=64, num_heads=8))
    # model.add(keras_nlp.layers.TransformerEncoder(intermediate_dim=64, num_heads=8))
    # model.add(keras_nlp.layers.TransformerEncoder(intermediate_dim=64, num_heads=8))
    # model.add(layers.Reshape((120,),input_shape=(6,20,)))

    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(6,5,4), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    #
    # # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))

    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Dense(num_words))
    # states = keras.layers.Input((-1,6,5,4))

    opt = tf.keras.optimizers.SGD(learning_rate=0.00001) #previously working
    # opt = keras.optimizers.SGD(learning_rate=0.000001) #previously working
    # opt = keras.optimizers.SGD(learning_rate=0.00000001) #previously working
    # opt = keras.optimizers.SGD(learning_rate=0.0001) #slightly worse
    # opt = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=opt, loss=loss_function)
    model.build()
    model.summary()
    return model


