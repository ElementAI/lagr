import copy
from typing import Optional
import re
import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.optim import AdamW, Adam
from collections import Counter
import networkx as nx
from scipy.optimize import linear_sum_assignment

from transformers import (
	AutoConfig,
	AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup, T5EncoderModel,T5Config
)

from src.utils import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, DEVICE, compare_alignments, add_eos
from src.utils.statistics import Metrics
from src.utils.graph_utils import GRAPH_PAD_TOKEN, NULL_TOKEN
from src.transformer import (
	TransformerEncDecModel, Transformer, 
	TransformerWithSeparateEncoderLAGr, 
	TransformerWithSharedEncoderLAGr
)

import pytorch_lightning as pl


def cross_entropy(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean", smoothing: float = 0,
				  ignore_index: Optional[int] = None) -> torch.Tensor:

	# Flatten inputs to 2D
	t2 = target.flatten().long()
	i2 = input.flatten(end_dim=-2)

	# If no smoothing, use built-in negative log loss
	if smoothing == 0:
		criterion = nn.NLLLoss(ignore_index=-100 if ignore_index is None else ignore_index, reduction=reduction)
		loss = criterion(i2, t2)
		if reduction == "none":
			return loss.view_as(target)
		else:
			return loss
	
	# Calculate the softmax cross entropy loss
	right_class = i2.gather(-1, t2.unsqueeze(-1)).squeeze()
	others = i2.sum(-1) - right_class

	# KL divergence
	loss = (smoothing - 1.0) * right_class - others * smoothing
	optimal_loss = -((1.0 - smoothing) * math.log(1 - smoothing) + (i2.shape[1] - 1) * smoothing * math.log(smoothing))

	loss = loss - optimal_loss

	# Handle masking if igonore_index is specified
	if ignore_index is not None:
		tmask = t2 != ignore_index
		loss = torch.where(tmask, loss, torch.zeros([1], dtype=loss.dtype, device=loss.device))
		n_total = tmask.float().sum()
	else:
		n_total = t2.nelement()

	# Reduction
	if reduction == "none":
		return loss.view_as(target)
	elif reduction == "mean":
		return loss.sum() / n_total
	elif reduction == "sum":
		return loss.sum()
	else:
		assert False, f"Invalid reduction {reduction}"


class Parser(pl.LightningModule):
	def __init__(self, args):
		super().__init__()
		self.save_hyperparameters()
		self._create_model(args)

	def _create_model(self, args):

		self.args = vars(args)
		self.data = args.data
		self.weakly = args.weak_supervision
		self.model_type = args.model_type
		self.flags = ['train_final', 'test', 'gen'] if self.data == 'cogs' else ['train_final', 'test']   # hacky but a way to name the test outputs

		self.k = args.k
		self.noise = args.noise
		self.stats = Metrics(args.preprocessor, args.tgt_vocab, decoder_sos_eos=args.tgt_vocab_size)
		self.graph_accuracy = self.stats.cfq_graph_accuracy if self.data == 'cfq' else self.stats.cogs_graph_accuracy
		# os.makedirs(os.path.join(self.data, args.log_dir), exist_ok=True)
		# self.saved_predictions = open(os.path.join(self.data, args.log_dir, args.log_file), mode='w+', encoding='utf-8')

		self.cached_alignments = {}
		self.expected_cache_size = args.expected_cache_size

		# Baseline Transformer
		if self.model_type == "transformer_baseline":
			model = Transformer
		elif self.model_type == "transformer_lagr":
			if self.args["share_encoder"]:
				model = TransformerWithSharedEncoderLAGr
			else:
				model = TransformerWithSeparateEncoderLAGr
		else: 
			raise Exception(f"{self.model_type} is an unknown model type.")

		self.model = TransformerEncDecModel(
			n_input_tokens=self.args["vocab_size"],
			n_out_tokens=self.args["tgt_vocab_size"],
			state_size=self.args["dim"],
			ff_multiplier=self.args["transformer.ff_multiplier"],
			nhead=self.args["transformer.n_heads"],
			num_encoder_layers=self.args["transformer.encoder_n_layers"],
			num_decoder_layers=self.args["transformer.decoder_n_layers"] or \
								self.args["transformer.encoder_n_layers"],
			dropout=self.args["dropout"],
			transformer=model,
			n_graph_layers=self.args['n_graph_layers'],
			n_node_labels=self.args['n_node_labels'],
			n_edge_labels=self.args['n_edge_labels'],
			tied_embedding=self.args["transformer.tied_embedding"],
			embedding_init="kaiming", 
			scale_mode="down"
		)

	def on_post_move_to_device(self):
		if self.args["share_layers"]:
			for mod in self.encoder.layers:
				mod.weight = self.encoder.layers[0].weight

	def configure_optimizers(self):
		# We will support Adam or AdamW as optimizers.
		if self.args["optimizer"]=="AdamW":
			opt = AdamW
		elif self.args["optimizer"]=="Adam":
			opt = Adam
		optimizer = opt(self.parameters(), **self.args["optimizer_args"])
		
		# We will reduce the learning rate by 0.1 after 100 and 150 epochs
		if self.args["scheduler"]=="linear_warmup":
			scheduler = get_linear_schedule_with_warmup(optimizer,**self.args["scheduler_args"])
		elif self.args["scheduler"]=="tf":
			scheduler = get_tf_schedule(optimizer)

		return {"optimizer":optimizer, "lr_scheduler":{"scheduler":scheduler,"interval":"step"}}

	def loss(self, outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, ignore_index: int) -> torch.Tensor:
		"""
		outputs: log probability scores
		targets: target tokens
		mask: bool mask with True for non-pad tokens
		"""
		l = cross_entropy(outputs, targets, reduction="none", smoothing=0, ignore_index=ignore_index)
		l = l.reshape_as(targets) * mask
		l = l.sum() / mask.sum()
		return l

	def _calculate_loss(self, batch):
		"""
		Runs model, and returns raw scores and loss
		"""
		src_tokens = add_eos(batch["src"], batch["src_len"], self.model.encoder_eos)
		tgt_tokens = add_eos(batch["tgt"], batch["tgt_len"], self.model.decoder_sos_eos)
		src_length_with_eos = batch["src_len"] + 1
		tgt_length_with_eos = batch["tgt_len"] + 1
		result = self.model(src_tokens, src_length_with_eos, tgt_tokens, tgt_length_with_eos, 
				teacher_forcing=self.training, max_len=tgt_length_with_eos.max().item())
		
		if self.model_type == "transformer_baseline":
			# assumes SL x BS shape
			max_len = tgt_tokens.shape[1]
			train_eos = True 
			len_mask = ~self.model.generate_len_mask(max_len, tgt_length_with_eos if train_eos else (tgt_length))
			result = F.log_softmax(result, -1)
			loss = self.loss(result, tgt_tokens, len_mask, ignore_index=PAD_TOKEN)

		# LAGr
		else:
			node_scores, edge_scores = result
			node_loss = self.loss(node_scores, batch["node_tgt"], batch["node_tgt"].ne(GRAPH_PAD_TOKEN), ignore_index=GRAPH_PAD_TOKEN)
			edge_loss = self.loss(edge_scores, batch["edge_tgt"], batch["edge_tgt"].ne(GRAPH_PAD_TOKEN), ignore_index=GRAPH_PAD_TOKEN)
			loss = node_loss + edge_loss

		return loss, result

	def training_step(self, batch, batch_idx):
		if self.weakly:
			loss, scores, preds, alignment_changes = self._weakly_sup_loss(batch)
			if alignment_changes >= 0: self.log("align_changes", alignment_changes, prog_bar=True)	# only track once it's trackable
		else:
			loss, scores = self._calculate_loss(batch)

		scheduler = self.lr_schedulers()
		self.log("train_loss", loss, prog_bar=True)
		
		return loss

	def infer_alignment(self, node_scores, edge_scores, node_targets, edge_targets, input_tokens, freeze_alignments=False):
		"""
		Args:
			node_scores: log scores (BS, SL, # node labels)
			edge_scores: log scores (BS, SL, SL, # of edge labels)
			node_targets: node labels (BS, SL)
			edge_targets: edge labels (BS, SL, SL)
			input_token = input sequences (BS x SL)

		Returns:
			Performs Maximum Flow Minimum cost using log probabilities to infer best alignment,
			then returns
			> aligned node_scores and edge_scores
			> node and edge predictions
			> loss
			> best_alignment
		"""
		batch_size, seq_len = node_targets.shape

		# Get scores for Bipartite Matching
		###################################
		# repeat each label for each input pos, i.e., along the 1st axis
		node_targets_tmp = torch.cat([node_targets.unsqueeze(1)] * seq_len, 1)   # BS, SL -> BS, SL, SL 

		# Selects predicted scores for true labels
		relevant_node_scores = torch.gather(node_scores, 2, node_targets_tmp)   # BS, SL, token_dim -> BS,  SL, SL

		# Perform maximum bipartite matching
		###################################
		k_alignments, k_alignment_changes, repeated_example_ids = self.find_k_maximum_matching(
			relevant_node_scores, input_tokens=input_tokens, node_targets=node_targets,
		)
		k_losses, k_node_preds, k_edge_preds = [], [], []

		# if repeated_example_ids:
		#     import ipdb;ipdb.set_trace()

		# Permute log probabilities with each alignment
		###################################
		for alignment in k_alignments:
			# NODES
			# node preds: permute node targets with alignment
			node_preds = torch.gather(node_targets, -1, alignment)
			aligned_node_scores = torch.gather(node_scores, -1, node_preds.unsqueeze(-1)).squeeze(-1)  # BS, SL, #tokens -> BS, SL

			# EDGES
			# edge label preds: permute edge targets with alignment
			align = torch.cat([alignment.unsqueeze(1)] * seq_len, 1)
			column_permuted = torch.gather(edge_targets, -1, align)
			edge_preds = torch.gather(column_permuted, 1, align.transpose(1, 2))
			aligned_edge_scores = torch.gather(edge_scores, -1, edge_preds.unsqueeze(-1)).squeeze(
				-1)  # BS, SL, SL, #tokens -> BS, SL, SL

			# LOSS UNDER INFERRED ALIGNMENT
			masked_node_losses = node_targets.ne(GRAPH_PAD_TOKEN) * aligned_node_scores
			masked_edge_losses = edge_targets.ne(GRAPH_PAD_TOKEN) * aligned_edge_scores
			loss = torch.sum(-masked_node_losses, axis=-1) + torch.sum(torch.sum(-masked_edge_losses, axis=-1), axis=-1)

			k_losses.append(loss)
			k_node_preds.append(node_preds)
			k_edge_preds.append(edge_preds)

		# Select alignment with the smallest cost
		##############
		best_k = torch.argmin(torch.stack(k_losses), axis=0)  # gives best index among the k alignments for each example (bs)
		best_loss = torch.mean(torch.gather(torch.stack(k_losses), 0, best_k.view(1,-1)).view(-1))
		best_alignments = torch.gather(
			torch.stack(k_alignments), 0, torch.stack(
				[best_k] * seq_len).transpose(0,1).view(1,-1, seq_len)    # bs, seq_len
		)
		best_node_preds = torch.gather(
			torch.stack(k_node_preds), 0, torch.stack(
				[best_k] * seq_len).transpose(0,1).view(1,-1, seq_len)
		).view(-1, seq_len)   # bs, seq_len

		best_edge_preds = torch.gather(
			torch.stack(k_edge_preds), 0, torch.stack(
				[best_k] * seq_len * seq_len).transpose(0,1).view(1,-1, seq_len, seq_len)
		).view(-1, seq_len, seq_len)   # bs, seq_len, seq_len

		# pick alignment changes for corresponding alignment
		if k_alignment_changes:
			actual_alignment_changes = torch.sum(
				torch.gather(torch.tensor(k_alignment_changes).to(DEVICE), 0, best_k.view(1,-1)))
			alignment_changes = actual_alignment_changes / float(batch_size)
		else:
			alignment_changes = -1.

		return (best_node_preds, best_edge_preds), best_loss, best_alignments, alignment_changes
	
	def _lookup_alignments(self, src):
		"""Looks up cached alignment for each input sequence"""
		batch_of_indices = []
		for sequence in src:
			example_key = tuple(sequence[sequence != PAD_TOKEN])
			alignment = self.cached_alignments.get(example_key, None)
			if not alignment:
				raise Exception(f"{example_key} still missing from cached alignments. This needs to be built first")
			batch_of_indices.append(alignment)

		alignment = torch.tensor(batch_of_indices)
		alignment_changes = [0.] * bs 
		return alignment, alignment_changes, 0

	def find_k_maximum_matching(self, relevant_node_scores, input_tokens=None, node_targets=None, \
		freeze_alignments=False):
		"""
		Estimates K approximate alignment candidates with noisy node scores.
		"""
		check_alignment_changes = False
		if len(self.cached_alignments) == self.expected_cache_size:
			check_alignment_changes = True

		if freeze_alignments: 
			return self._lookup_alignments(input_tokens)

		top_k_alignments = []
		k_alignment_changes = []

		for _ in range(self.k):
			# Adds noise to node scores
			noise = (torch.rand(relevant_node_scores.shape) * (self.noise)).to(DEVICE)
			noisy_scores = relevant_node_scores + noise
			
			# Calculate approx alignment with noisy scores
			alignment, alignment_changes, repeated_example_ids = self.find_maximum_matching(
				noisy_scores, input_tokens, node_targets, check_alignment_changes=check_alignment_changes)
			
			# Save alignment for later
			top_k_alignments.append(alignment.to(DEVICE))
			if check_alignment_changes:
				k_alignment_changes.append(alignment_changes)

		return top_k_alignments, k_alignment_changes, repeated_example_ids

	def find_maximum_matching(self, relevant_node_scores, src, node_targets, check_alignment_changes=False):
		"""
		Builds bipartite graph and finds the maximum matching, i.e. the best alignment that maximizes the log probabilities.
		Args:
			relevant_node_scores:  (BS x SL x SL)
			src: (BS x SL)
		Returns:
			batch of alignment indices where an index denotes the position in the target sequence.
		"""
		bs = node_targets.shape[0] 
		src = src.detach().cpu().numpy()

		batch_of_indices = []

		# num_layers = int(relevant_node_scores[0].shape[0] / src[0].shape[0])
		relevant_node_scores = relevant_node_scores.detach().cpu().numpy()
		node_targets = node_targets.detach().cpu().numpy()

		repeated_example_ids = []
		alignment_changes = []

		for example_id, cost in enumerate(relevant_node_scores):
			cost = relevant_node_scores[example_id].copy()
			node_targ_seq = node_targets[example_id]
			counter = Counter(node_targ_seq)
			repeated_labels_example = np.any([True for k,v in counter.items() if v > 1 and k != 0])
			if repeated_labels_example:
				repeated_example_ids.append(example_id)

			# example id for caching alignments and constraining
			in_sequence = src[example_id]
			# remove padding from inp sequence
			example_key = tuple(in_sequence[in_sequence != PAD_TOKEN])

			# relevant_positions = np.where((node_targ_seq != GRAPH_PAD_TOKEN) & (node_targ_seq != NULL_TOKEN))[0]
			relevant_positions = np.where(node_targ_seq != GRAPH_PAD_TOKEN)[0]
			
			non_pad_cost = cost[relevant_positions][:, relevant_positions]
			_, col_ids = linear_sum_assignment(-non_pad_cost)
			col_ids += 1 * self.args["n_graph_layers"]    # shift to right by n_graph_layers (due to padding for BOS token)

			all_col_ids = np.arange(node_targ_seq.shape[0]) 
			all_col_ids[relevant_positions] = col_ids     # keep nulls and pads at their original place, only permute actual nodes

			batch_of_indices.append(all_col_ids)

			# Calculate % of examples where alignment changes
			if check_alignment_changes:
				prev_alignment = self.cached_alignments[example_key]
				alignment_changes += compare_alignments(prev_alignment, col_ids, node_targ_seq)

			# Save latest alignment in cache
			self.cached_alignments[example_key] = col_ids

		alignment = torch.tensor(batch_of_indices)

		return alignment, alignment_changes, repeated_example_ids
	
	def _weakly_sup_loss(self, batch):
		# Adds EOS to src and tgt
		src_tokens = add_eos(batch["src"], batch["src_len"], self.model.encoder_eos)
		tgt_tokens = add_eos(batch["tgt"], batch["tgt_len"], self.model.decoder_sos_eos)
		src_length_with_eos = batch["src_len"] + 1
		tgt_length_with_eos = batch["tgt_len"] + 1
		
		# Runs model
		result = self.model(src_tokens, src_length_with_eos, tgt_tokens, tgt_length_with_eos, 
				teacher_forcing=self.training, max_len=tgt_length_with_eos.max().item())
		
		aligned_preds, loss, _, alignment_changes = self.infer_alignment(
			*result, batch["node_tgt"], batch["edge_tgt"], input_tokens=batch["src"])

		return loss, result, aligned_preds, alignment_changes

	def validation_step(self, batch, _, dataset_idx):
		if dataset_idx == 0:
			flag = 'val'
		else: 
			flag = 'train_val'
		self._eval_model(batch, flag=flag)

	def test_step(self,batch, _, dataset_idx):
		self._eval_model(batch, flag=self.flags[dataset_idx])
	
	def _eval_model(self, batch, flag):
		"""Evaluates current model on given batch of data. Returns loss, and accuracy."""

		if self.model_type == "transformer_baseline":
			loss, scores = self._calculate_loss(batch)
			exact_acc, acc = self.stats._compute_acc(batch["tgt"], batch["tgt_len"], scores)

			self.log(f"{flag}_exact_acc", exact_acc, prog_bar=True, add_dataloader_idx=False)
			self.log(f"{flag}_acc", acc, prog_bar=True, add_dataloader_idx=False)
			self.log(f"{flag}_loss", loss, prog_bar=True, add_dataloader_idx=False)	

		elif self.model_type == "transformer_lagr":
			if not self.weakly:
				loss, scores = self._calculate_loss(batch)
				node_acc, edge_acc, node_exact_acc, edge_exact_acc, exact_acc = self.stats.strongly_sup_metrics(batch, *scores)
				
				self.log(f"{flag}_loss", loss, prog_bar=True, add_dataloader_idx=False)
				self.log(f"{flag}_node_acc", node_acc, prog_bar=True, add_dataloader_idx=False)
				self.log(f"{flag}_edge_acc", edge_acc, prog_bar=True, add_dataloader_idx=False)
				self.log(f"{flag}_node_exact_acc", node_exact_acc, prog_bar=True, add_dataloader_idx=False)
				self.log(f"{flag}_edge_exact_acc", edge_exact_acc, prog_bar=True, add_dataloader_idx=False)
				self.log(f"{flag}_exact_acc", exact_acc, prog_bar=True, add_dataloader_idx=False)
			else:
				src_tokens = add_eos(batch["src"], batch["src_len"], self.model.encoder_eos)
				tgt_tokens = add_eos(batch["tgt"], batch["tgt_len"], self.model.decoder_sos_eos)
				# Runs model
				(node_scores, edge_scores) = self.model(src_tokens, batch["src_len"]+1, tgt_tokens,  batch["tgt_len"]+1,
									max_len=batch["tgt_len"].max().item()+1)

				# Argmax inference
				node_preds = torch.argmax(node_scores, axis=-1)
				edge_preds = torch.argmax(edge_scores, axis=-1)	

				# Graph accuracy - For CFQ only exact acc is available
				_,_,_,_, exact_acc = self.graph_accuracy(node_preds, edge_preds, batch)
				self.log(f"{flag}_exact_acc", exact_acc, prog_bar=True, add_dataloader_idx=False)

				# Restarts job if not converging on train graph accuracy
				if self.data == 'cogs' and flag == 'train_val' and exact_acc < .98 and self.global_step > 15000:
					print(f'Model not converging on training set w exact acc of {exact_acc} at step={self.global_step}.')
					sys.exit(1)
		else:
			raise Exception(f"Unknown model: {self.model_type}")

