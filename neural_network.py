import tensorflow as tf
import numpy as np
from typing import List, Union
import keras_nlp
import string
from state import ObservedState
from keras import Model

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

def qsa_error_loss_function(y_true, y_pred):
    """
    :param y_true: N x 2 tensor, concatenation of targets and actions
    :param y_pred: N x NUM_Words tensor
    :return:
    targets: N x 1
    targets - y_pred: N x Num_Words
    """
    qsa_prediction, _ = tf.split(y_pred, 2, axis=-1)
    targets, actions, _ = tf.split(y_true, num_or_size_splits=3, axis=-1)
    actions_one_hot = tf.one_hot(tf.cast(tf.squeeze(actions, axis=-1), tf.int32), depth=qsa_prediction.shape[-1])
    loss = tf.math.reduce_mean(
        tf.math.reduce_sum(tf.math.multiply(tf.math.pow(targets - qsa_prediction, 2), actions_one_hot), axis=-1),
        axis=-1)

    return loss

def word_prediction_loss_function(y_true,y_pred):
    """
    :param y_true: N x 2 tensor, concatenation of targets and actions
    :param y_pred: N x NUM_Words tensor
    :return:
    targets: N x 1
    targets - y_pred: N x Num_Words
    """
    _, word_idx_logits = tf.split(y_pred, 2, axis=-1)
    _, _, hidden_words_idx = tf.split(y_true,num_or_size_splits=3,axis=-1)

    scce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    word_prediction_scce_loss = scce_loss(hidden_words_idx, word_idx_logits)

    return word_prediction_scce_loss

def total_loss(y_true,y_pred):
    qsa_error_loss = qsa_error_loss_function(y_true, y_pred)
    word_prediction_loss = word_prediction_loss_function(y_true, y_pred)
    return qsa_error_loss + 1 * word_prediction_loss

def build_q_sa_model(num_words: int):
    x = tf.keras.layers.Input((6,5,num_features))
    y = tf.keras.layers.Reshape((6,5*num_features),input_shape=(6,5,num_features))(x)
    
    y = tf.keras.layers.Dense(32)(y)
    y = tf.keras.layers.LayerNormalization()(y)
    y = keras_nlp.layers.TransformerEncoder(intermediate_dim=64, num_heads=8)(y)
    # y = keras_nlp.layers.TransformerEncoder(intermediate_dim=64, num_heads=8)(y)
    # y = keras_nlp.layers.TransformerEncoder(intermediate_dim=64, num_heads=8)(y)

    y = tf.keras.layers.Flatten()(y)
    z = tf.keras.layers.Dense(128, activation='relu')(y)
    z = tf.keras.layers.LayerNormalization()(z)

    q_sa_head = tf.keras.layers.Dense(num_words)(z)
    predicted_word_head = tf.keras.layers.Dense(num_words)(z)

    output = tf.concat([q_sa_head, predicted_word_head], axis=-1)

    model = Model(inputs=x, outputs=output)

    # opt = tf.keras.optimizers.SGD(learning_rate=0.00001) #previously working
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001) #previously working


    model.compile(optimizer=opt, loss=total_loss, metrics=[qsa_error_loss_function, word_prediction_loss_function])
    # model.build()
    model.summary()
    return model


