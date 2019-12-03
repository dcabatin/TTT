import numpy as np
import tensorflow as tf
import numpy as np

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
##########DO NOT CHANGE#####################

def pad_corpus(french, english, lensent):
	"""
	DO NOT CHANGE:

	arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.

	:param french: list of French sentences
	:param english: list of English sentences
	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
	"""
	FRENCH_padded_sentences = []
	FRENCH_sentence_lengths = []
	for line in french:
		padded_FRENCH = line[:lensent]
		padded_FRENCH += [STOP_TOKEN]
		padded_FRENCH += [PAD_TOKEN] * (lensent - (len(padded_FRENCH)-1))
		FRENCH_padded_sentences.append(padded_FRENCH)

	ENGLISH_padded_sentences = []
	ENGLISH_sentence_lengths = []
	for line in english:
		padded_ENGLISH = line[:lensent]
		padded_ENGLISH = [START_TOKEN] + padded_ENGLISH + [STOP_TOKEN]
		padded_ENGLISH += [PAD_TOKEN] * (lensent - (len(padded_ENGLISH)-2))
		ENGLISH_padded_sentences.append(padded_ENGLISH)

	return FRENCH_padded_sentences, ENGLISH_padded_sentences

def build_vocab(sentences):
	"""
	DO NOT CHANGE

  Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE

  Convert sentences to indexed

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
	DO NOT CHANGE

  Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text


def get_data(french_file, english_file, lensent):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.

	:param french_training_file: Path to the french training file.
	:param english_training_file: Path to the english training file.
	:param french_test_file: Path to the french test file.
	:param english_test_file: Path to the english test file.

	:return: Tuple of train containing:
	(2-d list or array with english training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with english test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with french training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with french test sentences in vectorized/id form [num_sentences x 14]),
	english vocab (Dict containg word->index mapping),
	french vocab (Dict containg word->index mapping),
	english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""
	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index

	french, english = pad_corpus(read_data(french_file), read_data(english_file), lensent)

	french_vocab, _ = build_vocab(french)
	english_vocab, pad_token_idx = build_vocab(english)

	english = convert_to_id(english_vocab, english)
	french = convert_to_id(french_vocab, french)

	np.random.shuffle(english)
	np.random.shuffle(french)

	en_train_num = int(0.9*len(english))
	fr_train_num = int(0.9*len(french))

	return english[:en_train_num], english[en_train_num:], french[:fr_train_num], french[fr_train_num:], english_vocab, french_vocab, pad_token_idx
