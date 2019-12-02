import os
import numpy as np
import torch.nn as nn
import torch.functional as F
import torch
import numpy as np
from preprocess import *
from universal_transformer import UniversalTransformer
import sys

def get_loss(logits, labels, mask):
	# adapted from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
	# logits_flat: (batch * max_len, num_classes)
	logits_flat = logits.view(-1, logits.size(-1))
	# log_probs_flat: (batch * max_len, num_classes)
	log_probs_flat = F.log_softmax(logits_flat)
	# target_flat: (batch * max_len, 1)
	target_flat = labels.view(-1, 1)
	# losses_flat: (batch * max_len, 1)
	losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
	# losses: (batch, max_len)
	losses = losses_flat.view(*labels.size())
	losses = losses * mask.float()
	return losses.sum() / mask.float().sum().sum()

def get_accuracy(logits, labels, mask):
	decoded_symbols = np.argmax(logits, axis=2)
	correct = torch.eq(decoded_symbols, labels)
	correct_flat = correct.view(-1)
	mask_flat = mask.view(-1)
	accuracy = correct_flat[mask_flat.bool()].mean()
	return accuracy

def train(model, train_from_lang, train_to_lang, to_lang_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initilized model to use for forward and backward pass
	:param train_from_lang: from_lang train data (all data for training) of shape (num_sentences, 14)
	:param train_to_lang: to_lang train data (all data for training) of shape (num_sentences, 15)
	:param to_lang_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:return: None
	"""
	loss_layer = nn.CrossEntropyLoss()
	indices = np.array(range(train_from_lang.shape[0]))
	np.random.shuffle(indices)
	from_shuf = train_from_lang[indices, ...]
	to_shuf = train_to_lang[indices, ...]
	for i in range(0, train_from_lang.shape[0], model.batch_size):
		model.optimizer.zero_grad()
		from_data = from_shuf[i: i + model.batch_size]
		to_data = to_shuf[i: i + model.batch_size]
		logits = model.forward(from_data, to_data[:, :-1])
		labels = to_data[:, 1:]
		loss = get_loss(logits, labels, torch.ne(labels, to_lang_padding_index))
		loss.backward()
		model.optimizer.step()
		print('Train perplexity for batch of {}-{} / {} is {}'.format(i, i + model.batch_size, train_from_lang.shape[0], torch.exp(loss)))
		print('Loss is', loss)

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
	nonpad_correct = 0
	nonpad_seen = 0
	for i in range(0, test_from_lang.shape[0] - model.batch_size + 1, model.batch_size):
		from_data = test_from_lang[i: i + model.batch_size]
		to_data = test_to_lang[i: i + model.batch_size]
		prbs = model.forward(from_data, to_data[:, :-1])
		labels = to_data[:, 1:]
		mask = torch.ne(labels, to_lang_padding_index)
		total_loss += get_loss(prbs, labels, mask)
		np_seen_batch = mask.sum().sum()
		nonpad_seen += np_seen_batch
		nonpad_correct += np_seen_batch * get_accuracy(prbs, labels, mask)
		steps += 1

	return torch.exp(total_loss / steps), nonpad_correct / nonpad_seen

def main():
	print("Running preprocessing...")
	lensent = 13
	train_from_lang,test_from_lang,train_to_lang,test_to_lang,\
	from_lang_vocab,to_lang_vocab,to_lang_padding_index = \
		get_data('data/MTNT/train/train.fr-en.tsv', 'data/MTNT/test/test.fr-en.tsv', lensent)
	print("Preprocessing complete.")
	print('tl shape:', train_to_lang.shape, 'fl shape:', train_from_lang.shape)

	model_args = (train_from_lang.shape[1], train_to_lang.shape[1] - 1, len(from_lang_vocab), len(to_lang_vocab))
	model = UniversalTransformer(*model_args)


	# TODO:
	# Train and Test Model for n epochs
	n_epochs = 20
	for _ in range(n_epochs):
		train(model, train_from_lang, train_to_lang, to_lang_padding_index)
	perplexity, acc = test(model, test_from_lang, test_to_lang, to_lang_padding_index)
	print('Perplexity: ', perplexity)
	print('Accuracy: ', acc)
if __name__ == '__main__':
	main()


