import tensorflow as tf
from tensorflow import keras
import numpy as np

class Encoder(keras.Model):
    def __init__(self, units):
        super().__init__()
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
        self.W = keras.layers.Dense(w_units)
        self.W1 = keras.layers.Dense(w_units)
        self.v = keras.layers.Dense(1)
        self.tanh = keras.activations.tanh
        self.softmax = keras.activations.softmax
        
    def call(self, enc_hidden, dec_hidden):
        dec_hidden = tf.expand_dims(dec_hidden, 1)
        unnorm = self.v(
            self.tanh(self.W(enc_hidden) + self.W1(dec_hidden)))
        
        attention_weights = self.softmax(unnorm, axis=1)
        
        # Compute the context vector used to generate the decoder state
        c_vec = tf.reduce_sum(attention_weights * enc_hidden, axis=1)
        
        # Return the context vector and the pointer logits
        return c_vec, tf.squeeze(unnorm, axis=2), tf.squeeze(attention_weights, axis=2)  
    
class Decoder(keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.lstm = keras.layers.LSTM(self.units, return_state=True)
        
    def call(self, x, enc_out, prev_state):
        concatenated_inp = tf.concat([x, enc_out], axis=1)
        concatenated_inp = tf.expand_dims(concatenated_inp, 1)
        d, dec_h, dec_c =  self.lstm(concatenated_inp, initial_state=prev_state)
        return d, dec_h, dec_c
    

class PointerNetwork(keras.Model):
    def __init__(self,
                 enc_units,
                 dec_units, 
                 att_units, 
                 max_len, 
                 start_token, 
                 end_token):
        super().__init__()
        self.encoder = Encoder(enc_units)
        self.decoder = Decoder(dec_units)
        self.attention = Attention(att_units)
        self.embeddings = False
        self.max_len = max_len
        self.start_token = start_token
        self.end_token = end_token
        
        self.optimizer = tf.train.AdamOptimizer()
    
    def set_embeddings_layer(self, embeddings_layer):
        self.embeddings = embeddings_layer
    
    def predict_batch(self, X):
        assert self.embeddings, "Call self.set_embeddings_layer first"
        X = self.embeddings(X)
        enc_states, h1, h2 = self.encoder(X)
        input_tokens = tf.convert_to_tensor([self.start_token] * X.shape[0])
        enc_output = h1
        outputs = []
        
        for _ in range(self.max_len):
            dec_input = self.embeddings(input_tokens)
            decoded_state, h1, h2 = self.decoder(dec_input, enc_output, [h1, h2])
            enc_output, _, pointer_prob,  = self.attention(enc_states, 
                                                           decoded_state)
            # We decode with greedy search, to be done: Beam Search
            input_tokens = tf.argmax(pointer_prob, axis=1)
            outputs.append(input_tokens)
            
        return outputs
        
    def compute_loss(self, X, y):
        X = self.embeddings(X)
        enc_states, h1, h2 = self.encoder(X)
        enc_output = h1
        input_tokens = tf.convert_to_tensor([self.start_token] * X.shape[0])
        loss = 0
        for t in range(y.shape[1]):
            dec_input = self.embeddings(input_tokens)
            decoded_state, h1, h2 = self.decoder(dec_input, enc_output, [h1, h2])
            last_state, pointer_logits, _  = self.attention(enc_states, 
                                                            decoded_state)

            loss += tf.losses.sparse_softmax_cross_entropy(labels=y[:, t], 
                                                           logits=pointer_logits)
            input_tokens = y[:, t]
        
        # Dont forget to divide by summary lenght N, since we lose the /N component n by calling
        # N times softmax cross entropy
        loss = loss / int(y.shape[1])
        print('Train loss is', loss)
        return loss
            
    def train(self, X, y):
        assert self.embeddings, "Call self.load_embeddings first"
        self.optimizer.minimize(lambda: self.compute_loss(X, y))


