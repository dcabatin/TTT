from torch import F
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, MultiheadAttention, Module, Sequential
import torch.nn as nn

class UniversalTransformer(Module):
	def __init__(self, in_seq_len, out_seq_len, in_vocab_len, out_vocab_len, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.batch_size = 40
		self.nheads = 8
		self.encoder_T = 6
		self.decoder_T = 6
		self.embedding_size = 256
		self.dropout = 0.3
		self.in_seq_len = in_seq_len
		self.out_seq_len = out_seq_len

		self.enc_layer = UniversalTransformerEncoder(
			in_seq_len, self.nheads, self.encoder_T, self.dropout, self.embedding_size)
		self.dec_layer = UniversalTransformerDecoder(
			out_seq_len, self.nheads, self.decoder_T, self.dropout, self.embedding_size)
		self.enc_embedding_layer = nn.Embedding(in_vocab_len, self.embedding_size)
		self.dec_embedding_layer = nn.Embedding(out_vocab_len, self.embedding_size)

	def forward(self, encoder_input, decoder_input):
		enc_embeddings = self.enc_embedding_layer(encoder_input)
		enc_output = self.enc_layer(enc_embeddings)
		dec_embeddings = self.dec_embedding_layer(decoder_input)
		dec_output = self.dec_layer(dec_embeddings, enc_output)
		return dec_output

class UniversalTransformerEncoder(Module):
	def __init__(self, seq_len, nheads, T, dropout, emb_size, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.T = T
		layer_norm = nn.LayerNorm((seq_len, emb_size))
		self.enc = TransformerEncoder(TransformerEncoderLayer(emb_size, nheads, dim_feedforward=1024, dropout=dropout), 1, norm=layer_norm)

	def forward(self, x):
		for i in range(self.T):
			x = self.enc(x)
		return x


class UniversalTransformerDecoder(Module):
	def __init__(self, seq_len, nheads, T, dropout, emb_size, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.T = T
		layer_norm = nn.LayerNorm((seq_len, emb_size))
		self.dec = TransformerDecoder(TransformerDecoderLayer(emb_size, nheads, dim_feedforward=1024, dropout=dropout), 1, norm=layer_norm)

	def forward(self, x, encoder_out):
		for i in range(self.T):
			x = self.dec(x, encoder_out)
		return x

