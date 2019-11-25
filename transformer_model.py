import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters
		# 2) Define embeddings, encoder, decoder, and feed forward layers

		# Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 32
		self.learning_rate = 0.001
		self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

		# Define english and french embedding layers:
		self.fr_embedding_layer = tf.keras.layers.Embedding(french_vocab_size, self.embedding_size)
		self.flatten_layer = tf.keras.layers.Flatten()
		self.en_embedding_layer = tf.keras.layers.Embedding(english_vocab_size, self.embedding_size)

		# Create positional encoder layers
		self.fr_positional_layer = transformer.Position_Encoding_Layer(french_window_size, self.embedding_size)
		self.en_positional_layer = transformer.Position_Encoding_Layer(english_window_size, self.embedding_size)

		# Define encoder and decoder layers:
		self.encoder_layer = transformer.Transformer_Block(self.embedding_size, False)
		self.decoder_layer = transformer.Transformer_Block(self.embedding_size, True)

		# Define dense layer(s)
		self.dense_layer_1 = tf.keras.layers.Dense(64, activation='relu', use_bias=True)
		self.dense_layer_2 = tf.keras.layers.Dense(english_vocab_size, activation='softmax', use_bias=True)

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""

		# TODO:
		# 1) Add the positional embeddings to french sentence embeddings
		fr_embeddings = self.fr_embedding_layer(encoder_input)
		fr_embeddings = self.fr_positional_layer(fr_embeddings)
		# 2) Pass the french sentence embeddings to the encoder
		encoder_context = self.encoder_layer(fr_embeddings)
		# 3) Add positional embeddings to the english sentence embeddings
		en_embeddings = self.en_positional_layer(self.en_embedding_layer(decoder_input))
		# 4) Pass the english embeddings and output of your encoder, to the decoder
		decoder_out = self.decoder_layer(en_embeddings, encoder_context)
		# 3) Apply dense layer(s) to the decoder out to generate probabilities
		dense_out = self.dense_layer_2(self.dense_layer_1(decoder_out))
		return tf.reshape(dense_out, (-1, self.english_window_size, self.english_vocab_size))

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		# Note: you can reuse this from rnn_model.
		return tf.reduce_mean(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs), mask))

