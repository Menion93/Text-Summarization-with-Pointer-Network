import tensorflow as tf

class PointerDecoder:

  def __init__(self, embeddings, encoder, decoder, attention, pointer_switch,
                 max_len, start_token, end_token):
      self.encoder = encoder
      self.decoder = decoder
      self.attention = attention
      self.pointer_switch = pointer_switch
      self.embeddings = embeddings
      self.max_len = max_len
      self.start_token = start_token
      self.end_token = end_token

  def predict_batch(self, X):
      X = tf.convert_to_tensor(X)

      embed = self.embeddings(X)
      enc_states, h1, h2 = self.encoder(embed)
      input_tokens = tf.convert_to_tensor(
          [self.start_token] * embed.shape[0])
      # put last encoder state as attention vec at start
      c_vec = h1
      outputs = []

      for _ in range(self.max_len):
          dec_input = self.embeddings(input_tokens)
          decoded_state, h1, h2, decoded_probs = self.decoder(dec_input,
                                                              c_vec,
                                                              [h1, h2])
          c_vec, pointer_probs = self.attention(enc_states,
                                                decoded_state)

          # Compute switch probability to decide where to extract the next
          # word token
          switch_probs = self.pointer_switch(h1, c_vec)
          # Decode based on switch probs
          input_tokens = self.decode_next_word(switch_probs,
                                              decoded_probs,
                                              X,
                                              pointer_probs)
          outputs.append(input_tokens)

      return tf.transpose(tf.convert_to_tensor(outputs))


  def decode_next_word(self, switch_probs, decoded_probs, inputs, att_probs):
      sampled_probs = tf.random.uniform(switch_probs.shape, 0, 1)
      tokens = []
      token = None

      for prob, sampled, decoded, inp, att_p in zip(switch_probs,
                                                    sampled_probs,
                                                    decoded_probs,
                                                    inputs,
                                                    att_probs):
          if prob.numpy()[0] >= sampled.numpy()[0]:
              token = self.fixed_vocab_decode(decoded)
          else:
              token = self.pointer_greedy_search(att_p, inp)

          tokens.append(token)

      return tf.convert_to_tensor(tokens, dtype=tf.float32)


  def pointer_greedy_search(self, probs, inputs):
      return inputs[tf.argmax(probs)]


  def fixed_vocab_decode(self, decoded_probs):
      return tf.argmax(decoded_probs)
