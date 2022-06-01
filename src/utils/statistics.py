from typing import Optional
import re
import numpy as np
import torch
import networkx as nx
from collections import Counter

from src.utils import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, DEVICE, compare_alignments, add_eos
from src.utils.graph_utils import GRAPH_PAD_TOKEN, NULL_TOKEN


def split_and_sort_clauses(lambd):
	return sorted(re.split(" AND | ; ", lambd))


class Metrics:

	def __init__(self, preprocessor, tgt_vocab, decoder_sos_eos):

		self.preprocessor = preprocessor
		self.tgt_vocab = tgt_vocab
		self.decoder_sos_eos = decoder_sos_eos

	def cogs_graph_accuracy(self, node_preds, edge_preds, batch):
		lmbd_tgt, node_tgt, edge_tgt = batch["tgt"], batch["node_tgt"], batch["edge_tgt"]

		# Node Accuracy
		node_results = node_preds.eq(node_tgt)[node_tgt != GRAPH_PAD_TOKEN]
		n_correct_nodes = node_results.sum().item()
		n_nodes = float(node_results.size(0))
		n_exact_node_matches = (
			node_preds.eq(node_tgt)
			.bitwise_or(node_tgt == GRAPH_PAD_TOKEN)
			.all(1).sum().item()
		)

		# Edge Accuracy
		edge_results = edge_preds.eq(edge_tgt)[edge_tgt != GRAPH_PAD_TOKEN]
		n_correct_edges = edge_results.sum().item()
		n_edges = float(edge_results.size(0))
		n_exact_edge_matches = (
			edge_preds.eq(edge_tgt)
			.bitwise_or(edge_tgt == GRAPH_PAD_TOKEN)
			.all(1).all(1).sum().item()
		)
		n_examples = float(lmbd_tgt.shape[0])

		n_correct_lmbd = self.lambda_accuracy(node_preds, edge_preds, lmbd_tgt, node_tgt, edge_tgt)

		return (
			n_correct_nodes / n_nodes,
			n_correct_edges / n_edges,
			n_exact_node_matches / n_examples,
			n_exact_edge_matches / n_examples,
			n_correct_lmbd / n_examples,
		)

	def _serialized_graph(self, nodes, edges):
		triples = []
		for node, n_edges in zip(nodes, edges):
			# if node in [NULL_TOKEN, GRAPH_PAD_TOKEN]:
			# 	continue

			agents = n_edges.eq(self.preprocessor.edge_vocab.stoi['agent'])
			predicates = nodes[agents]     # predicate of the given subject
			positions = torch.where(agents)[0]
			for idx, pred in zip(positions, predicates):
				pred_edges = edges[idx,:]   # get destination edges for predicate node
				destination_nodes = nodes[pred_edges.eq(self.preprocessor.edge_vocab.stoi['theme'])]
				if destination_nodes.shape[0] > 0:
					for dest in destination_nodes:
						triples.append(' '.join([
								self.preprocessor.node_vocab.itos[node],
								self.preprocessor.node_vocab.itos[pred],
								self.preprocessor.node_vocab.itos[dest]])
						)
				else:
					triples.append(' '.join([
							self.preprocessor.node_vocab.itos[node],
							self.preprocessor.node_vocab.itos[pred]])
					)
			# check for filters
			filter_clauses = nodes[n_edges.eq(self.preprocessor.edge_vocab.stoi['FILTER'])]     # predicate of the given subject
			for filter_node in filter_clauses:
				triples.append('FILTER ' + ' '.join([
								self.preprocessor.node_vocab.itos[node],
								'!=',
								self.preprocessor.node_vocab.itos[filter_node]])
				)
		return triples
	
	def _create_graph(self, nodes, edges, pad_mask, true=False):
		# take edge predictions for valid nodes
		mask = nodes.ne(NULL_TOKEN) & nodes.ne(GRAPH_PAD_TOKEN) & pad_mask
		valid_nodes = nodes[mask]
		edges_for_predicted_nodes = edges[mask][:, mask]

		# destination nodes for each predicted node
		edge_positions = torch.where(edges_for_predicted_nodes.ne(NULL_TOKEN) & edges_for_predicted_nodes.ne(GRAPH_PAD_TOKEN))  # non null or padding
		predicted_labels = edges_for_predicted_nodes[edge_positions].detach().cpu().numpy()

		triples = self._serialized_graph(valid_nodes, edges_for_predicted_nodes)

		src_nodes = valid_nodes[edge_positions[0]].detach().cpu().numpy()
		dest_nodes = valid_nodes[edge_positions[1]].detach().cpu().numpy()

		G = nx.DiGraph()
		for node1, node2, label in zip(src_nodes, dest_nodes, predicted_labels):
			if node1 not in [NULL_TOKEN, GRAPH_PAD_TOKEN] and node2 not in [NULL_TOKEN, GRAPH_PAD_TOKEN]:
				G.add_edge(node1, node2, label=label)
			else:
				# should never be 0 or 1
				raise Exception('should never be 0 or 1')

		return G, sorted(triples)

	def cfq_graph_accuracy(self, node_preds, edge_preds, batch, out_file=None):
		src, node_tgt, edge_tgt = batch["src"].detach().cpu().numpy(), batch["node_tgt"], batch["edge_tgt"]

		n_correct_graphs = 0.
		n_correct_nodes, n_correct_edges = 0., 0.
		n_nodes, n_edges = 0., 0.
		n_graphs = len(node_preds)

		for nodes, edges, true_nodes, true_edges, inputs in zip(node_preds, edge_preds, node_tgt, edge_tgt, src):
			pad_mask = true_nodes.ne(GRAPH_PAD_TOKEN)
			G_pred, serialized_pred = self._create_graph(nodes, edges, pad_mask, true=False)
			G_true, serialized_true = self._create_graph(true_nodes, true_edges, pad_mask, true=True)

			tmp = [e for e in G_pred.edges(data=True) if e in G_true.edges(data=True)]
			n_correct_nodes += len([n for n in G_pred.nodes if n in G_true.nodes])
			n_correct_edges += len([e for e in G_pred.edges(data=True) if e in G_true.edges(data=True)])
			n_nodes += len(G_true.nodes)
			n_edges += len(G_true.edges)

			if not nx.is_empty(G_pred) and not nx.is_empty(G_true):
				nm = nx.algorithms.isomorphism.categorical_edge_match('label','label')
				if nx.is_isomorphic(G_pred, G_true, edge_match=nm):
					n_correct_graphs += 1.
				else:
					if out_file:
						question_str = ' '.join([self.src_vocab.itos[int(n)] for n in inputs if self.src_vocab.itos[int(n)] != '<blank>'])
						inp_seq = inputs.repeat(2).view(-1, inputs.shape[0]).transpose(0,1).reshape(-1).detach().cpu().numpy()

						redundant_nodes = ' '.join([self.preprocessor.node_vocab.itos[e] for e in G_pred.nodes if e not in G_true.nodes])
						missing_nodes = ' '.join([self.preprocessor.node_vocab.itos[e] for e in G_true.nodes if e not in G_pred.nodes])
						redundant_edges = self._get_str([e for e in G_pred.edges.data() if e not in G_true.edges.data()])
						missing_edges = self._get_str([e for e in G_true.edges.data() if e not in G_pred.edges.data()])
						
						out_file.write('QUESTION : ' + question_str + '\n')
						if missing_nodes:
							out_file.write('MISSING nodes: \n' + missing_nodes + '\n')
						if redundant_nodes:
							out_file.write('REDUNDANT nodes: \n' + redundant_nodes + '\n')

						if missing_edges:
							out_file.write('MISSING edges: \n' + missing_edges + '\n')
						if redundant_edges:
							out_file.write('REDUNDANT edges: \n' + redundant_edges + '\n')

						# predicted_nodes_str = ' '.join([self.preprocessor.node_vocab.itos[n] for n in nodes[nodes.ne(0) & nodes.ne(1)]])
						# out_file.write('PREDICTED_NODES: \n' + predicted_nodes_str + '\n')
						out_file.write('PREDICTED: \n' + '\n'.join(serialized_pred) + '\n')
						out_file.write('ACTUAL:    \n' + '\n'.join(serialized_true) + '\n')
						out_file.write('INP to PRED: ' + ' '.join([
							self.src_vocab.itos[int(n)] + ':' + self.preprocessor.node_vocab.itos[e] 
							for n, e in zip(inp_seq, nodes) if self.src_vocab.itos[int(n)] != '<blank>'
						]) + '\n')
						out_file.write('###############\n')
						out_file.flush()

		exact_acc = n_correct_graphs / n_graphs
		return n_correct_nodes/n_nodes, n_correct_edges/n_edges, -1., -1., exact_acc
	
	def _get_str(self, edges):
		return ' '.join([
			' '.join([
				self.preprocessor.node_vocab.itos[e[0]],
				self.preprocessor.edge_vocab.itos[e[2]['label']],
				self.preprocessor.node_vocab.itos[e[1]]
			]) for e in edges]) 

	def strongly_sup_metrics(self, batch, node_scores, edge_scores):
		node_preds = torch.argmax(node_scores, -1)
		edge_preds = torch.argmax(edge_scores, -1)
		
		# Node, edge, exact lambda accuracies
		node_acc, edge_acc, node_exact_acc, edge_exact_acc, lmbda_exact_acc = self.cogs_graph_accuracy(node_preds, edge_preds, batch)

		return node_acc, edge_acc, node_exact_acc, edge_exact_acc, lmbda_exact_acc

	def lambda_accuracy(self, node_preds, edge_preds, lmbd_tgt, node_tgt, edge_tgt):
		"""Calculates if the predicted graph matches the exact lambda calculus """

		node_sets, edge_sets, edge_labels = self.preprocessor.scores_to_graph(
			node_preds, edge_preds, node_tgt, edge_tgt)

		# graph to lambda str expressions
		predicted_lambdas = self.preprocessor.graph_to_lambda(node_sets, edge_sets, edge_labels)

		n_exact_lambda_match = 0

		# iter through each example
		for batch_idx in range(lmbd_tgt.shape[0]):

			pred = predicted_lambdas[batch_idx]
			tgt = lmbd_tgt[batch_idx].view(-1)
			tgt = tgt[tgt != PAD_TOKEN]

			# convert target tokens to lambda form
			true_lambda = []
			for token in tgt:
				word = self.tgt_vocab.itos[token.item()]
				if word in ["<s>", "</s>", "<blank>"]:
					continue
				else:
					true_lambda.append(word)
			true_lambda = " ".join(true_lambda)

			# primitive type questions won"t be serialized into a lambda. These will be considered correct
			# if graph is correct
			if pred == "NA":
				# check if primitive example
				all_nodes_true = torch.all(node_preds[batch_idx].eq(node_tgt[batch_idx]).bitwise_or(node_tgt[batch_idx] == GRAPH_PAD_TOKEN))
				all_edges_true = torch.all(edge_preds[batch_idx].eq(edge_tgt[batch_idx]).bitwise_or(edge_tgt[batch_idx] == GRAPH_PAD_TOKEN))

				if all_nodes_true and all_edges_true:
					n_exact_lambda_match += 1

			if split_and_sort_clauses(true_lambda) == split_and_sort_clauses(pred):
				n_exact_lambda_match += 1

		return n_exact_lambda_match

	def _compute_acc(self, target, tgt_lengths, scores):
		predictions = scores.max(-1)[1]
		target = add_eos(target, tgt_lengths, self.decoder_sos_eos)
		
		mask = target.eq(PAD_TOKEN)  # bool
		
		correct_tokens = predictions.eq(target).masked_select(~mask)
		n_correct_tokens = correct_tokens.sum().item()
		acc = n_correct_tokens / correct_tokens.size(0)

		correct_sequences = (predictions.eq(target) | mask).all(1)  # correct sequences
		num_correct_sequence = correct_sequences.sum().item()
		exact_acc = num_correct_sequence / correct_sequences.size(0)

		# tokens it missed to predict correctly
		# c = Counter(target.masked_select(target.ne(predictions)).cpu().numpy())

		return exact_acc, acc