from tensorflow import keras


def load_pretrained_embeddings(embed_matrix, trainable=False):
    return keras.layers.Embedding(embed_matrix.shape[0],
                                  embed_matrix.shape[1],
                                  weights=[embed_matrix],
                                  trainable=trainable)


def get_new_embeddings(voc_len, embedding_dim):
    return keras.layers.Embedding(voc_len, embedding_dim)
