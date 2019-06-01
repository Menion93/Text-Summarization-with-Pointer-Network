import tensorflow as tf
from tensorflow import keras
import numpy as np


class Encoder(keras.Model):
    def __init__(self, units):
        super(Encoder, self).__init__()
        self.units = units
        self.lstm = keras.layers.LSTM(self.units,
                                      return_sequences=True,
                                      return_state=True)

    def call(self, x):
        sequences, state1, state2 = self.lstm(x)
        return sequences, state1, state2


class Attention(keras.Model):
    def __init__(self, w_units):
        super().__init__()
        self.W = keras.layers.Dense(w_units, use_bias=False)
        self.W1 = keras.layers.Dense(w_units, use_bias=False)
        self.v = keras.layers.Dense(1)
        self.tanh = keras.activations.tanh
        self.softmax = keras.activations.softmax

    def call(self, enc_hidden, dec_hidden):
        dec_hidden = tf.expand_dims(dec_hidden, 1)
        unnorm = self.v(
            self.tanh(self.W(enc_hidden) + self.W1(dec_hidden))
        )

        attention_weights = self.softmax(unnorm, axis=1)

        # Compute the context vector used to generate the decoder state
        c_vec = tf.reduce_sum(attention_weights * enc_hidden, axis=1)

        # Return the context vector and the pointer logits and the pointer probs
        return c_vec, tf.squeeze(attention_weights, axis=2)


class Decoder(keras.Model):
    def __init__(self, units, output_size):
        super().__init__()
        self.units = units
        self.lstm = keras.layers.LSTM(self.units, return_state=True)
        self.output_layer = keras.layers.Dense(
            output_size, activation='softmax')

    def call(self, x, enc_out, prev_state):
        # Concatenate encoder output (or context vector) and the target/predicted embedding
        concatenated_inp = tf.concat([x, enc_out], axis=1)
        concatenated_inp = tf.expand_dims(concatenated_inp, 1)
        # Compute the hidden state h^d
        d, dec_h, dec_c = self.lstm(concatenated_inp, initial_state=prev_state)

        # Decode using vocabulary
        flattened = tf.layers.flatten(d)
        decoded_probs = self.output_layer(flattened)

        # Return Decode hidden states and vocabulary logits over the vocabulary
        return d, dec_h, dec_c, decoded_probs


class PointerSwitch(keras.Model):
    def __init__(self, units):
        super().__init__()
        self.W1 = keras.layers.Dense(units,  use_bias=False)
        self.W2 = keras.layers.Dense(units, use_bias=False)
        self.v = keras.layers.Dense(1)

    def call(self, enc, c_vec):
        '''
            Compute switch probabilities from the context vector and
            the encoder last output state
        '''
        switch_prob = tf.keras.activations.sigmoid(
            self.v(
                self.W1(enc) + self.W2(c_vec)
            )
        )

        return switch_prob
