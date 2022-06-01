import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable, Dict
from dataclasses import dataclass

from src.attention import MultiHeadAttention, AttentionMask


ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


def sinusoidal_pos_embedding(d_model: int, max_len: int = 5000, pos_offset: int = 0,
							 device: Optional[torch.device] = None):
	pe = torch.zeros(max_len, d_model, device=device)
	position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1) + pos_offset
	div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
	pe[:, 0::2] = torch.sin(position * div_term)
	pe[:, 1::2] = torch.cos(position * div_term)
	return pe


class PositionalEncoding(torch.nn.Module):
	r"""Inject some information about the relative or absolute position of the tokens
		in the sequence. The positional encodings have the same dimension as
		the embeddings, so that the two can be summed. Here, we use sine and cosine
		functions of different frequencies.
	.. math::
		\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
		\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
		\text{where pos is the word position and i is the embed idx)
	Args:
		d_model: the embed dim (required).
		dropout: the dropout value (default=0.1).
		max_len: the max. length of the incoming sequence (default=5000).
		batch_first: if true, batch dimension is the first, if not, its the 2nd.
	Examples:
		>>> pos_encoder = PositionalEncoding(d_model)
	"""

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False,
				 scale: float = 1):
		super(PositionalEncoding, self).__init__()
		self.dropout = torch.nn.Dropout(p=dropout)

		pe = sinusoidal_pos_embedding(d_model, max_len, 0) * scale

		self.batch_dim = 0 if batch_first else 1
		pe = pe.unsqueeze(self.batch_dim)

		self.register_buffer('pe', pe)

	def get(self, n: int, offset: int) -> torch.Tensor:
		return self.pe.narrow(1 - self.batch_dim, start=offset, length=n)

	def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
		x = x + self.get(x.size(1 - self.batch_dim), offset)
		return self.dropout(x)


class TiedEmbedding(torch.nn.Module):
	def __init__(self, weights: torch.Tensor):
		super().__init__()

		# Hack: won't save it as a parameter
		self.w = [weights]
		self.bias = torch.nn.Parameter(torch.zeros(self.w[0].shape[0]))

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		return F.linear(t, self.w[0], self.bias)


class TransformerEncoderLayer(torch.nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu):
		super(TransformerEncoderLayer, self).__init__()
		self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
		self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
		self.dropout = torch.nn.Dropout(dropout)
		self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

		self.norm1 = torch.nn.LayerNorm(d_model)
		self.norm2 = torch.nn.LayerNorm(d_model)
		self.dropout1 = torch.nn.Dropout(dropout)
		self.dropout2 = torch.nn.Dropout(dropout)

		self.activation = activation
		self.reset_parameters()

	def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		src2 = self.self_attn(src, src, AttentionMask(mask, None))
		src = src + self.dropout1(src2)
		src = self.norm1(src)
		src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		src = src + self.dropout2(src2)
		src = self.norm2(src)
		return src

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
									  if self.activation is F.relu else 1.0)
		torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerDecoderLayer(torch.nn.Module):
	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation: ActivationFunction = F.relu):
		super(TransformerDecoderLayer, self).__init__()

		self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
		self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
		self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
		self.dropout = torch.nn.Dropout(dropout)
		self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

		self.norm1 = torch.nn.LayerNorm(d_model)
		self.norm2 = torch.nn.LayerNorm(d_model)
		self.norm3 = torch.nn.LayerNorm(d_model)
		self.dropout1 = torch.nn.Dropout(dropout)
		self.dropout2 = torch.nn.Dropout(dropout)
		self.dropout3 = torch.nn.Dropout(dropout)

		self.activation = activation
		self.reset_parameters()

	def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
				memory_key_padding_mask: Optional[torch.Tensor] = None,
				full_target: Optional[torch.Tensor] = None, pos_offset: int = 0) -> torch.Tensor:
		
		assert pos_offset == 0 or tgt_mask is None
		tgt2 = self.self_attn(tgt, tgt if full_target is None else full_target, mask=AttentionMask(None, tgt_mask))
		tgt = tgt + self.dropout1(tgt2)
		tgt = self.norm1(tgt)
		tgt2 = self.multihead_attn(tgt, memory, mask=AttentionMask(memory_key_padding_mask, None))
		tgt = tgt + self.dropout2(tgt2)
		tgt = self.norm2(tgt)
		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout3(tgt2)
		tgt = self.norm3(tgt)
		return tgt

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.linear1.weight, gain=torch.nn.init.calculate_gain('relu')
									  if self.activation is F.relu else 1.0)
		torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerDecoderBase(torch.nn.Module):
	@dataclass
	class State:
		step: int
		state: Dict[int, torch.Tensor]

	def __init__(self, d_model: int):
		super().__init__()
		self.d_model = d_model

	def create_state(self, batch_size: int, max_length: int, device: torch.device) -> State:
		return self.State(0, {i: torch.empty([batch_size, max_length, self.d_model], device=device)
							  for i in range(len(self.layers))})

	def one_step_forward(self, state: State, data: torch.Tensor, *args, **kwargs):
		assert data.shape[1] == 1, f"For one-step forward should have one timesteps, but shape is {data.shape}"
		assert state.step < state.state[0].shape[1]

		for i, l in enumerate(self.layers):
			state.state[i][:, state.step:state.step + 1] = data
			data = l(data, *args, **kwargs, full_target=state.state[i][:, :state.step + 1],
					 pos_offset=state.step)

		state.step += 1
		return data


class TransformerEncoder(torch.nn.Module):
	def __init__(self, layer, n_layers: int, *args, **kwargs):
		super().__init__()
		self.layers = torch.nn.ModuleList([layer(*args, **kwargs) for _ in range(n_layers)])

	def forward(self, data: torch.Tensor, *args, **kwargs):
		for l in self.layers:
			data = l(data, *args, **kwargs)
		return data


class TransformerDecoder(TransformerDecoderBase):
	def __init__(self, layer, n_layers: int, d_model: int, *args, **kwargs):
		super().__init__(d_model)
		self.layers = torch.nn.ModuleList([layer(d_model, *args, **kwargs) for _ in range(n_layers)])

	def forward(self, data: torch.Tensor, *args, **kwargs):
		for l in self.layers:
			data = l(data, *args, **kwargs)
		return data


def TransformerEncoderWithLayer(layer=TransformerEncoderLayer):
	return lambda *args, **kwargs: TransformerEncoder(layer, *args, **kwargs)


def TransformerDecoderWithLayer(layer=TransformerDecoderLayer):
	return lambda *args, **kwargs: TransformerDecoder(layer, *args, **kwargs)


class Transformer(torch.nn.Module):
	def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
				 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
				 activation: ActivationFunction = F.relu, encoder_layer=TransformerEncoderWithLayer(),
				 decoder_layer=TransformerDecoderWithLayer(), **kwargs):
		super().__init__()

		self.encoder = encoder_layer(num_encoder_layers, d_model, nhead, dim_feedforward,
									 dropout, activation)
		self.decoder = decoder_layer(num_decoder_layers, d_model, nhead, dim_feedforward,
									 dropout, activation)

	def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
				src_length_mask: Optional[torch.Tensor] = None):

		memory = self.encoder(src, src_length_mask)
		return self.decoder(tgt, memory, tgt_mask, src_length_mask)

	def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
		return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


class TransformerWithSeparateEncoderLAGr(nn.Module):
	def __init__(self, d_model: int = 512, n_graph_layers: int = 1, n_node_labels: int = 646, n_edge_labels: int = 11, 
				nhead=4, num_encoder_layers: int = 6, dim_feedforward=2048, dropout: float = 0.4, 
				activation: ActivationFunction = F.relu, encoder_layer=TransformerEncoderWithLayer(), **kwargs):
		super().__init__()
		
		self.node_encoder = encoder_layer(num_encoder_layers, d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
			dropout=dropout, activation=F.relu)
		self.edge_encoder = encoder_layer(num_encoder_layers, d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
			dropout=dropout, activation=F.relu)

		self.decoder = LAGr(n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, dim=d_model, n_graph_layers=n_graph_layers)
	
	def forward(self, src: torch.Tensor, src_length_mask: Optional[torch.Tensor] = None):
		node_embeds = self.node_encoder(src, src_length_mask)
		edge_embeds = self.edge_encoder(src, src_length_mask)
		node_scores, edge_scores = self.decoder(node_embeds, edge_embeds)

		return (node_scores, edge_scores)


class TransformerWithSharedEncoderLAGr(nn.Module):
	def __init__(self, d_model: int = 512, n_graph_layers: int = 1, n_node_labels: int = 646, n_edge_labels: int = 11, 
				nhead: int = 8, num_encoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
				activation: ActivationFunction = F.relu, encoder_layer=TransformerEncoderWithLayer(), **kwargs):
		super().__init__()

		self.encoder = encoder_layer(num_encoder_layers, d_model, nhead, dim_feedforward, dropout, activation=activation)
		self.decoder = LAGr(n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, dim=d_model, n_graph_layers=n_graph_layers)

	def forward(self, src: torch.Tensor, src_length_mask: Optional[torch.Tensor] = None):

		memory = self.encoder(src, src_length_mask)
		node_embeds, edge_embeds = memory, memory
		node_scores, edge_scores = self.decoder(node_embeds, edge_embeds)
		return (node_scores, edge_scores)


class TransformerEncDecModel(nn.Module):
	def __init__(self, n_input_tokens: int, n_out_tokens: int, state_size: int = 512, ff_multiplier: float = 4,
				 max_len: int=5000, transformer = Transformer, n_graph_layers: int=1, n_node_labels: int=646,
				 n_edge_labels: int=11, tied_embedding: bool=False,
				 pos_embeddig: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
				 encoder_sos: bool = True, same_enc_dec_embedding: bool = False, embedding_init: str = "pytorch",
				 in_embedding_size: Optional[int] = None, out_embedding_size: Optional[int] = None, 
				 scale_mode: str = "none", **kwargs):
		'''
		Transformer encoder-decoder.
		:param n_input_tokens: Number of channels for the input vectors
		:param n_out_tokens: Number of channels for the output vectors
		:param state_size: The size of the internal state of the transformer
		'''
		super().__init__()

		assert scale_mode in ["none", "opennmt", "down"]
		assert embedding_init in ["pytorch", "xavier", "kaiming"]

		assert (not same_enc_dec_embedding) or (n_input_tokens == n_out_tokens)

		self.tied_embedding = tied_embedding

		self.decoder_sos_eos = n_out_tokens   # weird name seems to be the EOS token for the decoder
		self.encoder_eos = n_input_tokens
		self.encoder_sos = n_input_tokens + 1 if encoder_sos else None
		self.state_size = state_size
		self.embedding_init = embedding_init
		
		self.n_graph_layers = n_graph_layers
		self.n_node_labels = n_node_labels
		self.n_edge_labels = n_edge_labels

		self.ff_multiplier = ff_multiplier
		self.n_input_tokens = n_input_tokens
		self.n_out_tokens = n_out_tokens
		self.in_embedding_size = in_embedding_size
		self.out_embedding_size = out_embedding_size
		self.same_enc_dec_embedding = same_enc_dec_embedding
		self.scale_mode = scale_mode
		self.pos = pos_embeddig or PositionalEncoding(state_size, max_len=max_len, batch_first=True,
										scale=(1.0 / math.sqrt(state_size)) if scale_mode == "down" else 1.0)
		
		self.register_buffer('int_seq', torch.arange(max_len, dtype=torch.long))
		self.construct(transformer, **kwargs)
		self.reset_parameters()

	def pos_embed(self, t: torch.Tensor, offset: int, scale_offset: int) -> torch.Tensor:
		if self.scale_mode == "opennmt":
			t = t * math.sqrt(t.shape[-1])
		
		return self.pos(t, offset)

	def construct(self, transformer, **kwargs):
		self.input_embedding = nn.Embedding(self.n_input_tokens + 1 + int(self.encoder_sos is not None), 
												  self.in_embedding_size or self.state_size)
		self.output_embedding = self.input_embedding if self.same_enc_dec_embedding else \
								nn.Embedding(self.n_out_tokens+1, self.out_embedding_size or self.state_size)

		if self.in_embedding_size is not None:
			self.in_embedding_upscale = nn.Linear(self.in_embedding_size, self.state_size)
		
		if self.out_embedding_size is not None:
			self.out_embedding_upscale = nn.Linear(self.out_embedding_size, self.state_size)

		if self.tied_embedding:
			assert self.out_embedding_size is None
			self.output_map = TiedEmbedding(self.output_embedding.weight)
		else:
			self.output_map = nn.Linear(self.state_size, self.n_out_tokens+1)

		self.trafo = transformer(
			d_model=self.state_size, dim_feedforward=int(self.ff_multiplier*self.state_size),
			n_graph_layers=self.n_graph_layers, n_node_labels=self.n_node_labels, 
			n_edge_labels=self.n_edge_labels, **kwargs)

	def reset_parameters(self):
		if self.embedding_init == "xavier":
			nn.init.xavier_uniform_(self.input_embedding.weight)
			nn.init.xavier_uniform_(self.output_embedding.weight)
		elif self.embedding_init == "kaiming":
			nn.init.kaiming_normal_(self.input_embedding.weight)
			nn.init.kaiming_normal_(self.output_embedding.weight)

		if not self.tied_embedding:
			nn.init.xavier_uniform_(self.output_map.weight)

	def generate_len_mask(self, max_len: int, len: torch.Tensor) -> torch.Tensor:
		return self.int_seq[: max_len] >= len.unsqueeze(-1)

	def output_embed(self, x: torch.Tensor) -> torch.Tensor:
		o = self.output_embedding(x)
		if self.out_embedding_size is not None:
			o = self.out_embedding_upscale(o)
		return o

	def run_greedy(self, src: torch.Tensor, src_len: torch.Tensor, max_len: int) -> torch.Tensor:
		batch_size = src.shape[0]
		n_steps = src.shape[1]

		in_len_mask = self.generate_len_mask(n_steps, src_len)
		memory = self.trafo.encoder(src, mask=in_len_mask)

		running = torch.ones([batch_size], dtype=torch.bool, device=src.device)
		out_len = torch.zeros_like(running, dtype=torch.long)

		next_tgt = self.pos_embed(self.output_embed(
			torch.full([batch_size,1], self.decoder_sos_eos, dtype=torch.long, device=src.device)
		), 0, 1)

		all_outputs = []
		state = self.trafo.decoder.create_state(src.shape[0], max_len, src.device)

		for i in range(max_len):
			output = self.trafo.decoder.one_step_forward(state, next_tgt, memory, memory_key_padding_mask=in_len_mask)

			output = self.output_map(output)
			all_outputs.append(output)

			out_token = torch.argmax(output[:,-1], -1)    # BS x prediction for given seq_len position
			running &= out_token != self.decoder_sos_eos   # BS x 1

			out_len[running] = i + 1
			next_tgt = self.pos_embed(self.output_embed(out_token).unsqueeze(1), i+1, 1)

		return torch.cat(all_outputs, 1)

	def run_teacher_forcing(self, src: torch.Tensor, src_len: torch.Tensor, target: torch.Tensor,
							target_len: torch.Tensor) -> torch.Tensor:
		# Adds SOS token to target sequence
		target = self.output_embed(F.pad(target[:, :-1], (1, 0), value=self.decoder_sos_eos).long())
		target = self.pos_embed(target, 0, 1)

		in_len_mask = self.generate_len_mask(src.shape[1], src_len)

		res = self.trafo(src, target, src_length_mask=in_len_mask,
						  tgt_mask=self.trafo.generate_square_subsequent_mask(target.shape[1], src.device))

		return self.output_map(res)

	def input_embed(self, x: torch.Tensor) -> torch.Tensor:
		src = self.input_embedding(x.long())
		if self.in_embedding_size is not None:
			src = self.in_embedding_upscale(src)
		return src

	def forward(self, src: torch.Tensor, src_len: torch.Tensor, target: torch.Tensor,
				target_len: torch.Tensor, teacher_forcing: bool = False, max_len: Optional[int] = None) -> torch.Tensor:
		'''
		Run transformer encoder-decoder on some input/output pair
		:param src: source tensor. Shape: [N, S], where S in the in sequence length, N is the batch size
		:param src_len: length of source sequences. Shape: [N], N is the batch size
		:param target: target tensor. Shape: [N, S], where T in the in sequence length, N is the batch size
		:param target_len: length of target sequences. Shape: [N], N is the batch size
		:param teacher_forcing: use teacher forcing or greedy decoding
		:param max_len: overwrite autodetected max length. Useful for parallel execution
		:return: prediction of the target tensor. Shape [N, T, C_out]
		'''

		if self.encoder_sos is not None:
			# Adds SOS to src
			src = F.pad(src, (1, 0), value=self.encoder_sos)
			src_len = src_len + 1
			
		src = self.pos_embed(self.input_embed(src), 0, 0)
		
		# if transformer baseline
		if not isinstance(self.trafo.decoder, LAGr):

			if teacher_forcing:
				return self.run_teacher_forcing(src, src_len, target, target_len)
			else:
				return self.run_greedy(src, src_len, max_len or target_len.max().item())

		# if transformer lagr 
		else:
			in_len_mask = self.generate_len_mask(src.shape[1], src_len)  # True for padding
			return self.trafo(src, src_length_mask=in_len_mask)


class LAGr(nn.Module):

	def __init__(self, n_node_labels=644+2, n_edge_labels=9+2, dim=512, n_graph_layers=1):
		"""
		Args:
			encoder: Chosen encoder used for both node and edge predictions.
			n_node_labels: Node label vocabulary size.
			n_edge_labels: Edge label vocabulary size.
			dim: Encoder dimension.
			n_graph_layers: Number of graph layers to use.
		"""
		super(LAGr, self).__init__()

		self.dim = dim

		self.n_node_labels = n_node_labels
		self.n_edge_labels = n_edge_labels
		self.n_graph_layers = n_graph_layers
		self.grapy_layer_transform = nn.Identity()

		self.head_dim = self.dim // self.n_edge_labels
		self.out_dim = self.head_dim * self.n_edge_labels

		self.node_model = nn.Linear(self.n_graph_layers * self.dim, self.n_graph_layers * self.n_node_labels)
		self.linear_keys = nn.Linear(self.n_graph_layers * self.dim, self.n_graph_layers * self.out_dim)
		self.linear_query = nn.Linear(self.n_graph_layers * self.dim, self.n_graph_layers * self.out_dim)

	def forward(self, node_embeds, edge_embeds):
		node_log_softmax, edge_log_softmax = self._forward(node_embeds, edge_embeds)

		return (node_log_softmax, edge_log_softmax)
	
	def _predict_nodes(self, node_embeds):
		"""
		Args:
			node_embeds: Node embeddings, i.e., output of node encoder model.
		Returns:
			node_log_softmax: softmax scores over all node labels.
		"""
		bs, seq_len, _ = node_embeds.shape

		out = self.node_model(node_embeds)      # [bs, seq_len, n_graph_layers x n_node_labels]
		out = out.view(bs, -1, self.n_node_labels)   # [bs, seq_len * n_graph_layers, n_node_labels]

		node_log_softmax = nn.LogSoftmax(dim=-1)(out)
		return node_log_softmax

	def _predict_edges(self, edge_embeds):
		"""
		Args:
			edge_embeds: Edge embeddings, i.e., output of edge encoder model.
		Returns:
			edge_log_softmax: softmax scores over all edge labels.
		"""
		bs, seq_len, _ = edge_embeds.shape
		new_length = seq_len*self.n_graph_layers

		key = self.linear_keys(edge_embeds)   # [bs, seq_len, n_graph_layers * n_heads * n_edge_labels]
		key = key.view(bs, -1, self.n_edge_labels, self.head_dim)  # [bs, seq_len * n_graph_layers, n_edge_labels, n_heads]
		key = key.transpose(1, 2).transpose(2, 3)  # [bs, n_edge_labels, n_heads, seq_len * n_graph_layers]

		query = self.linear_query(edge_embeds)
		query = query.view(bs, -1, self.n_edge_labels, self.head_dim)
		query = query.transpose(1, 2)   # [bs, n_edge_labels, seq_len * n_graph_layers, n_heads]

		# switch num_heads, heads_dim in keys
		dot = torch.matmul(query, key)
		assert dot.shape == torch.Size([bs, self.n_edge_labels, new_length, new_length])

		dot = dot.transpose(1, 3)
		edge_log_softmax = nn.LogSoftmax(dim=-1)(dot)  # [bs, new_length, new_length, n_edge_labels]
		return edge_log_softmax
	
	def _forward(self, node_embeds, edge_embeds):
		# Populates graph layers
		node_embeds, edge_embeds = self._interleave(node_embeds, edge_embeds)

		# Graph predictions
		node_log_softmax = self._predict_nodes(node_embeds)
		edge_log_softmax = self._predict_edges(edge_embeds)

		return (node_log_softmax, edge_log_softmax)

	def _interleave(self, node_embeds, edge_embeds):
		"""
		Takes node and edge embeddings of dim [seq_len, bs, dim], and populates all graph layers.
		Currently, done by repeating the same embedding. E.g. [a,b,c] x 2 -> [a,a,b,b,c,c]
		"""
		bs, seq_len, dim = node_embeds.shape
		node_embeds = node_embeds.repeat(1, 1, self.n_graph_layers).view(bs, seq_len, dim*self.n_graph_layers)
		edge_embeds = edge_embeds.repeat(1, 1, self.n_graph_layers).view(bs, seq_len, dim*self.n_graph_layers)
		
		return node_embeds, edge_embeds