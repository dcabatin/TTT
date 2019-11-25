import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
import sys


def train(model, train_from_lang, train_to_lang, to_lang_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initilized model to use for forward and backward pass
	:param train_from_lang: from_lang train data (all data for training) of shape (num_sentences, 14)
	:param train_to_lang: to_lang train data (all data for training) of shape (num_sentences, 15)
	:param to_lang_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:return: None
	"""

	indices = tf.random.shuffle(range(train_from_lang.shape[0]))
	fr_shuf = tf.gather(train_from_lang, indices)
	en_shuf = tf.gather(train_to_lang, indices)
	for i in range(0, train_from_lang.shape[0], model.batch_size):
		fr_data = fr_shuf[i: i + model.batch_size]
		en_data = en_shuf[i: i + model.batch_size]
		with tf.GradientTape() as tape:
			prbs = model.call(fr_data, en_data[:, :-1])
			labels = en_data[:, 1:]
			loss = model.loss_function(prbs, labels, tf.not_equal(labels, to_lang_padding_index))
			print('Train perplexity for batch of {}-{} / {} is {}'.format(i, i + model.batch_size, train_from_lang.shape[0], tf.exp(loss)))
			print('Loss is', loss)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	# NOTE: For each training step, you should pass in the from_lang sentences to be used by the encoder,
	# and to_lang sentences to be used by the decoder
	# - The to_lang sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP]
	#
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]

def test(model, test_from_lang, test_to_lang, to_lang_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initilized model to use for forward and backward pass
	:param test_from_lang: from_lang test data (all data for testing) of shape (num_sentences, 14)
	:param test_to_lang: to_lang test data (all data for testing) of shape (num_sentences, 15)
	:param to_lang_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:returns: perplexity of the test set, per symbol accuracy on test set
	"""
	# Note: Follow the same procedure as in train() to construct batches of data!
	total_loss = 0
	total_acc = 0
	steps = 0
	for i in range(0, test_from_lang.shape[0], model.batch_size):
		fr_data = test_from_lang[i: i + model.batch_size]
		en_data = test_to_lang[i: i + model.batch_size]
		prbs = model.call(fr_data, en_data[:, :-1])
		labels = en_data[:, 1:]
		mask = tf.equal(labels, to_lang_padding_index)
		total_loss += model.loss_function(prbs, labels, mask)
		total_acc += model.accuracy_function(prbs, labels, mask)
		steps += 1

	return tf.exp(total_loss / steps), total_acc / steps

def main():
	print("Running preprocessing...")
	lensent = 13
	train_from_lang,test_from_lang,train_to_lang,test_to_lang,\
	from_lang_vocab,to_lang_vocab,to_lang_padding_index = \
		get_data('data/MTNT/train/train.fr-en.tsv', 'data/MTNT/test/test.fr-en.tsv', lensent)
	print("Preprocessing complete.")
	print('tl shape:', train_to_lang.shape, 'fl shape:', train_from_lang.shape)

	model_args = (train_from_lang.shape[1], len(from_lang_vocab), train_to_lang.shape[1] - 1, len(to_lang_vocab))
	model = Transformer_Seq2Seq(*model_args)


	# TODO:
	# Train and Test Model for 1 epoch.
	train(model, train_from_lang, train_to_lang, to_lang_padding_index)
	perplexity, acc = test(model, test_from_lang, test_to_lang, to_lang_padding_index)
	print('Perplexity: ', perplexity)
	print('Accuracy: ', acc)
if __name__ == '__main__':
   main()


