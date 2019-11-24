import numpy as np
import tensorflow as tf
import numpy as np

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
##########DO NOT CHANGE#####################

def pad_corpus(l_from_sents, l_to_sents, sent_len):
	"""
	DO NOT CHANGE:

	arguments are lists of from-lang, to-lang sentences. Returns [from-lang sents, to-lang sents]. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.

	:param l_from_sents: list of from-lang sentences
	:param l_to_sents: list of to-lang sentences
	:return: A tuple of: (list of padded sentences for from-lang, list of padded sentences for to-lang)
	"""
	l_from_padded_sentences = []
	for line in l_from_sents:
		padded_l_from = line[:sent_len]
		padded_l_from += [STOP_TOKEN] + [PAD_TOKEN] * (sent_len - len(padded_l_from))
		l_from_padded_sentences.append(padded_l_from)

	l_to_padded_sentences = []
	for line in l_to_sents:
		padded_l_to = line[:sent_len]
		padded_l_to = [START_TOKEN] + padded_l_to + [STOP_TOKEN] + [PAD_TOKEN] * (sent_len - len(padded_l_to))
		l_to_padded_sentences.append(padded_l_to)

	return l_from_padded_sentences, l_to_padded_sentences


def build_vocab(sentences):
	"""
	DO NOT CHANGE

    Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
    """
	tokens = []
	for s in sentences:
		tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN, PAD_TOKEN, UNK_TOKEN] + tokens)))

	vocab = {word: i for i, word in enumerate(all_words)}

	return vocab, vocab[PAD_TOKEN]


def convert_to_id(vocab, sentences):
	"""
    Convert sentences to indexed

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	print([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence]  for sentence in sentences])
	return np.stack(
		[[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_parallel_data(file_name):
	"""
    Load text data from parallel file

	:param file_name:  string, name of data file
	:return: [from_lang sentences, to_lang sentences] where each is a list of lists of words
  """
	l_from_sents = []
	l_to_sents = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file:
			l_from, l_to = line.split('\t')[1:3]
			l_from_sents.append(l_from.split())
			l_to_sents.append(l_to.split())
	return [l_from_sents, l_to_sents]


def get_data(train_file, test_file, sent_len=13):
	"""
	Get parallel data from given files and vectorize it.

	:param train_file: Path to the parallel training file.
	:param test_file: Path to the parallel testing file.
	:param sent_len: Max length of sentence (discounting start/stop tokens and padding).

	:return: Tuple containing:
	Vectorized train sentences in from-lang [num_sentences x (sent_len+1)],
	Vectorized test sentences in from-lang [num_sentences x (sent_len+1)],
	Vectorized train sentences in to-lang [num_sentences x (sent_len+2)],
	Vectorized test sentences in to-lang [num_sentences x (sent_len+2)],
	from-lang vocab (Dict containg word->index mapping),
	to-lang vocab (Dict containg word->index mapping),
	to-lang padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""
	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index

	# TODO:

	# 1) Read l_toglish and Frl_toch Data for training and testing (see read_data)
	l_from_train, l_to_train = read_parallel_data(train_file)
	l_from_test, l_to_test = read_parallel_data(test_file)

	# 2) Pad training data (see pad_corpus)
	l_from_train, l_to_train = pad_corpus(l_from_train, l_to_train, sent_len)
	# 3) Pad testing data (see pad_corpus)
	l_from_test, l_to_test = pad_corpus(l_from_test, l_to_test, sent_len)
	# 4) Build vocab for l_froml_toch (see build_vocab)
	l_from_vocab, _ = build_vocab(l_from_train)
	# 5) Build vocab for l_toglish (see build_vocab)
	l_to_vocab, pad_id = build_vocab(l_to_train)
	# 6) Convert training and testing l_toglish sl_totl_toces to list of IDS (see convert_to_id)
	l_to_train_ids = convert_to_id(l_to_vocab, l_to_train)
	l_to_test_ids = convert_to_id(l_to_vocab, l_to_test)
	# 7) Convert training and testing l_froml_toch sl_totl_toces to list of IDS (see convert_to_id)
	l_from_train_ids = convert_to_id(l_from_vocab, l_from_train)
	l_from_test_ids = convert_to_id(l_from_vocab, l_from_test)
	return l_from_train_ids, l_from_test_ids, l_to_train_ids, l_to_test_ids, l_from_vocab, l_to_vocab, pad_id