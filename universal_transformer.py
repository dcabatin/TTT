import torch
import torch.functional as F
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, MultiheadAttention, Module, Sequential
import torch.nn as nn
import math

class UniversalTransformer(Module):
	def __init__(self, in_seq_len, out_seq_len, in_vocab_len, out_vocab_len, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.batch_size = 200
		self.nheads = 4
		self.encoder_T = 6
		self.decoder_T = 6
		self.embedding_size = 32
		self.dropout = 0.3
		self.in_seq_len = in_seq_len
		self.out_seq_len = out_seq_len

		self.enc_layer = UniversalTransformerEncoder(
			in_seq_len, self.nheads, self.encoder_T, self.dropout, self.embedding_size)
		self.dec_layer = UniversalTransformerDecoder(
			out_seq_len, self.nheads, self.decoder_T, self.dropout, self.embedding_size)
		self.enc_embedding_layer = nn.Embedding(in_vocab_len, self.embedding_size)
		self.dec_embedding_layer = nn.Embedding(out_vocab_len, self.embedding_size)
		self.ff_layer = nn.Linear(self.embedding_size, out_vocab_len)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

	def forward(self, encoder_input, decoder_input):
		enc_embeddings = self.enc_embedding_layer(encoder_input)
		enc_output = self.enc_layer(enc_embeddings)
		dec_embeddings = self.dec_embedding_layer(decoder_input)
		dec_output = self.dec_layer(dec_embeddings, enc_output)
		return self.ff_layer(dec_output)

class UniversalTransformerEncoder(Module):
	def __init__(self, seq_len, nheads, T, dropout, emb_size, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.T = T
		layer_norm = nn.LayerNorm((seq_len, emb_size))
		self.enc = TransformerEncoder(TransformerEncoderLayer(emb_size, nheads, dim_feedforward=1024, dropout=dropout), 1, norm=layer_norm)
		self.pos = PositionalTimeEncoding(emb_size, seq_len)

	def forward(self, x):
		for i in range(self.T):
			x = self.pos(x, i)
			x = self.enc(x)
		return x


class UniversalTransformerDecoder(Module):
	def __init__(self, seq_len, nheads, T, dropout, emb_size, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.T = T
		layer_norm = nn.LayerNorm((seq_len, emb_size))
		self.dec = TransformerDecoder(TransformerDecoderLayer(emb_size, nheads, dim_feedforward=1024, dropout=dropout), 1, norm=layer_norm)
		self.pos = PositionalTimeEncoding(emb_size, seq_len)

	def forward(self, x, encoder_out):
		for i in range(self.T):
			x = self.pos(x, i)
			x = self.dec(x, encoder_out)
		return x

class PositionalTimeEncoding(nn.Module):
	"Implement the PE function."

	def __init__(self, d_model, seq_len):
		super().__init__()
		self.seq_len = seq_len
		self.d_model = d_model

		# Compute the positional encodings once in log space.
		self.pe = torch.zeros(seq_len, d_model)
		position = torch.arange(0, seq_len).unsqueeze(1)
		self.div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

		self.pe[:, 0::2] = torch.sin(position * self.div_term)
		self.pe[:, 1::2] = torch.cos(position * self.div_term)
		self.pe = self.pe.unsqueeze(0)

	def forward(self, x, t):
		device = x.device
		x = x + self.pe[:, :x.size(1)].to(device)
		pt = torch.zeros(self.seq_len, self.d_model).to(device)
		pt[:, 0::2] = torch.sin(t * self.div_term.to(device))
		pt[:, 1::2] = torch.cos(t * self.div_term.to(device))
		x = x + pt
		return x

