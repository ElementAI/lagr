""" 
	Utility functions used to preprocess the COGS and CFQ datasets. 
"""

import yaml
import os
from typing import DefaultDict
import numpy as np
import json
import pandas as pd
from datetime import datetime

from types import SimpleNamespace

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from src.utils.graph_utils import (
	Preprocessor, NULL_TOKEN, GRAPH_PAD_TOKEN, 
	_cogs_build_graph, _cfq_build_graph
)


# Relevant filenames in COGS
COGS_DATASET_FILENAMES = ['train', 'test', 'dev', 'gen', 'gen_dev']
CFQ_DATASET_FILENAMES = ['train', 'test', 'validation']

# Special tokens
PAD_TOKEN = 0
EOS_TOKEN = 1
BOS_TOKEN = 2
UNK_TOKEN = 3
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNK_TOKEN]

PAD = '[PAD]'
EOS = '[EOS]'
BOS = '[BOS]'
UNK = '[UNK]'
DEVICE = 'cuda'


def _create_name(cl_args):
	params = f'bs{cl_args.batch_size}_lr{cl_args.lr}_dp{cl_args.dropout}_dim{cl_args.dim}_ep{cl_args.epochs}'

	if cl_args.model_type == 'transformer_baseline':
		run_name = f'seed:{cl_args.seed}_{cl_args.model_type}' + params
		exp_name = f'{cl_args.data}_{cl_args.model_type}'
		return exp_name, run_name

	if cl_args.share_encoder:
		encoder_type = 'shared'
	else:
		encoder_type = 'separate'
	if cl_args.weak_supervision:
		supervision='weakly_sup'
	else:
		supervision='strongly_sup'
	run_name = f'seed:{cl_args.seed}_{cl_args.split}_{supervision}_{encoder_type}_{cl_args.model_type}_k={cl_args.k}_noise={cl_args.noise}' + params
	exp_name = f'{cl_args.data}_{cl_args.split}_{supervision}_{encoder_type}_{cl_args.model_type}_k={cl_args.k}_noise={cl_args.noise}'

	return exp_name, run_name


def _ckpt_callback(cl_args, exp_name, run_name, possible_run_id):
	if possible_run_id:
		run_id = possible_run_id
	else:
		run_id = datetime.now().strftime("%m%d%Y:%H:%M:%S")
	dirpath = os.path.join(cl_args.data_path, 'ckpts', exp_name, run_name, run_id)

	return pl.callbacks.model_checkpoint.ModelCheckpoint(
		dirpath=dirpath,
		monitor='val_exact_acc',
		mode='max',
		save_last=True,
		filename='best_{val_exact_acc:02f}_at_{epoch:02d}',
		save_top_k=1,
		auto_insert_metric_name=True,
		verbose=True
	)
	

def make_collator():
	def collate_batch(data):

		batched = {}
		src_tensors, tgt_tensors, node_tgt_tensors, edge_tgt_tensors, src_len, tgt_len = pad_tensors(data)
		batched['src'] = src_tensors
		batched['tgt'] = tgt_tensors
		batched['node_tgt'] = node_tgt_tensors
		batched['edge_tgt'] = edge_tgt_tensors
		batched['src_len'] = src_len
		batched['tgt_len'] = tgt_len
		return batched

	return collate_batch


def _maybe_fetch_ckpt(cl_args, exp_name, run_name):
	if cl_args.ckpt:
		return cl_args.ckpt, ''

	try:
		ckpt_dir = os.path.join(cl_args.data_path, 'ckpts', exp_name, run_name)
		runs = os.listdir(ckpt_dir)
		runs.sort(key=lambda date: datetime.strptime(date, ("%m%d%Y:%H:%M:%S")))
		last_run_dir = runs[0]
		last_ckpt = os.path.join(ckpt_dir, last_run_dir, 'last.ckpt')
		print(f'Found previous ckpt under {ckpt_dir}.')
		return last_ckpt, last_run_dir
	except:
		print(f'Didn\'t find checkpoint under {ckpt_dir}')
		return '', ''


def parse_args(parser):
	cl_args = parser.parse_args()
	# Overwrite cl_args from config
	if cl_args.from_config:
		with open(cl_args.from_config, "r") as f:
			config_dict = yaml.safe_load(f)
		for arg, val in config_dict.items():
			setattr(cl_args, arg, val)
	return cl_args


def set_random_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	seed_everything(seed=seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def setup_data(cl_args):
	"""
	Splits source and target, builds graph targets and saves source and target vocabularies.
	"""

	if 'cogs' in cl_args.data_path:
		graph_path = cl_args.data_path + '/graph'
		input_path = cl_args.data_path + '/lambda'
		output_path = cl_args.data_path + '/lambda'
		files = COGS_DATASET_FILENAMES
		build_graph = _cogs_build_graph

	elif 'cfq' in cl_args.data_path:
		cfq_dir = os.path.join(cl_args.data_path, cl_args.split)
		graph_path = os.path.join(cfq_dir, 'graph')
		input_path = cfq_dir
		output_path = cfq_dir
		files = CFQ_DATASET_FILENAMES
		build_graph = _cfq_build_graph
	else:
		raise Exception(f'Parent path must contain cfq or cogs, but got {cl_args.data_path}.')

	# Create a new directory if the specified output path does not exist.
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	
	source_vocab = DefaultDict(int, {PAD: PAD_TOKEN}) #, EOS: EOS_TOKEN, UNK: UNK_TOKEN})
	target_vocab = DefaultDict(int, {PAD: PAD_TOKEN}) #, EOS: EOS_TOKEN, UNK: UNK_TOKEN})
	cur_src_vocab_iter = 1 #max(SPECIAL_TOKENS) + 1
	cur_tgt_vocab_iter = 1 #max(SPECIAL_TOKENS) + 1

	datasets = DefaultDict()

	# preprocessor to buildgraph meaning representations
	preprocessor = Preprocessor(data_dir=graph_path, graph_layers=cl_args.n_graph_layers)

	for filename in files:

		with open(os.path.join(input_path, f'{filename}.tsv')) as f:
			data = f.readlines()

		# Corresponding graph data
		file_path = os.path.join(graph_path, f'{filename}_parsed.json')
		graph = json.load(open(file_path))
		graph_df = pd.DataFrame(graph)
		src_to_graph_dict = graph_df.set_index('original_inp').agg(dict, 1).to_dict()

		source_lines = []
		target_lines = []

		src_tensors, tgt_tensors = [], []
		node_tgt_tensors, edge_tgt_tensors = [], []

		for line in data:
			source, target, _ = line.rstrip('\n').split('\t')
			source_lines.append('{}\n'.format(source))
			target_lines.append('{}\n'.format(target))

			# split sequences
			split_source = source.split()
			split_target = target.split()

			# builds tensor graph targets
			graph = src_to_graph_dict[source]
			input_length = len(split_source) * cl_args.n_graph_layers
			node_tgt_tensor, edge_tgt_tensor = build_graph(graph, preprocessor, input_length, n_special_tokens=2)

			node_tgt_tensors.append(node_tgt_tensor)
			edge_tgt_tensors.append(edge_tgt_tensor)

			# vocabs and tokenize
			src_tensor = []
			for w in split_source:
				# populate vocab dict
				if w not in source_vocab:
					source_vocab[w] = cur_src_vocab_iter
					cur_src_vocab_iter += 1
				# tokenize sentence
				src_tensor.append(source_vocab[w])

			src_tensors.append(src_tensor) # + [source_vocab[EOS]])   # adds EOS in the end

			tgt_tensor = []
			for w in split_target:
				# populate vocab dict
				if w not in source_vocab:
					source_vocab[w] = cur_src_vocab_iter
					cur_src_vocab_iter += 1
				# tokenize sentence
				tgt_tensor.append(source_vocab[w])

			tgt_tensors.append(tgt_tensor) # + [target_vocab[EOS]])   # adds EOS in the end
		
		dataset = SimpleNamespace()
		# Sequence source and targets
		dataset.src = src_tensors
		dataset.tgt = tgt_tensors
		# Graph targets
		dataset.node_tgt = node_tgt_tensors
		dataset.edge_tgt = edge_tgt_tensors
		datasets[filename] = dataset

		# Write the datapoints to source and target files.
		with open(os.path.join(output_path, f'{filename}_source.txt'), 'w') as wf:
			wf.writelines(source_lines)

		with open(os.path.join(output_path, f'{filename}_target.txt'), 'w') as wf:
			wf.writelines(target_lines)
		
	source_words = list(source_vocab.keys())
	# target_words = list(target_vocab.keys())

	# Write the vocabulary files.
	with open(os.path.join(output_path, 'source_vocab.txt'), 'w') as wf:
		for w in list(source_words):
			wf.write(w)
			wf.write('\n')

	# with open(os.path.join(output_path, 'target_vocab.txt'), 'w') as wf:
	# 	for w in list(target_words):
	# 		wf.write(w)
	# 		wf.write('\n')

	print(f'Reformatted and saved data to {output_path}.')

	return datasets, [Vocab(source_vocab), Vocab(source_vocab)], preprocessor


def add_eos(input: torch.Tensor, lengths: torch.Tensor, eos_id: int):
	input = torch.cat((input, torch.zeros_like(input[:,0:1])), dim=-1)
	input.scatter_(-1, lengths.unsqueeze(-1).long(), value=eos_id)
	return input


def pad_tensors(batch):
	"""
	batch: list of items where each item contains a src, tgt, node_tgt, edge_tgt.
	returns:
		padded tensors containing src, tgt, node_tgt, edge_tgt
	"""
	# Pad src and tgt sequences
	x = pad_sequence([example['src'] for example in batch], batch_first=True, padding_value=PAD_TOKEN)
	y = pad_sequence([example['tgt'] for example in batch], batch_first=True, padding_value=PAD_TOKEN)

	# Pad graph nodes
	node_targets = [example['node_tgt'] for example in batch]
	edge_targets = [example['edge_tgt'] for example in batch]
	node_tgt = pad_sequence(node_targets, batch_first=True, padding_value=GRAPH_PAD_TOKEN)
	max_len = node_tgt.shape[-1]

	# Pad graph edges
	res = []
	for e in edge_targets:
		tmp = torch.LongTensor([GRAPH_PAD_TOKEN] * max_len * max_len).view(max_len, max_len)
		tmp[:e.shape[0], :e.shape[1]] = e
		res.append(tmp)

	edge_tgt = torch.stack(res)

	src_len = torch.sum(x != PAD_TOKEN, 1)
	tgt_len = torch.sum(y != PAD_TOKEN, 1)

	return x, y, node_tgt, edge_tgt, src_len, tgt_len


def compare_alignments(prev_align, cur_align, node_tgt):
	"""
	Compares previous and current alignments to detect if the alignment has changed (1) for nodes
	that are not null or pad or not (0).
	"""
	true_node_positions = sorted(np.where(
		(node_tgt != NULL_TOKEN) & (node_tgt != GRAPH_PAD_TOKEN))[0])
	
	equal = True
	for e in true_node_positions:
		cur_pos = np.where(cur_align == e)[0][0]
		prev_pos = np.where(prev_align == e)[0][0]
		if cur_pos != prev_pos:
			equal = False
			return [1.]
	return [0.]


class Vocab():
	def __init__(self, vocab_dict):
		self.stoi = vocab_dict
		self.itos = {idx: word for word, idx in vocab_dict.items()}
	
	def __len__(self):
		return len(self.stoi)


class CustomDataset(Dataset):

	def __init__(self, dataset):
		self.src_tokens = dataset.src
		self.tgt_tokens = dataset.tgt
		self.node_tgt_tokens = dataset.node_tgt
		self.edge_tgt_tokens = dataset.edge_tgt

	def __len__(self):
		return len(self.src_tokens)   # length of a batch of data

	def __getitem__(self,index):

		item = {
			'src': torch.LongTensor(self.src_tokens[index]),
			'tgt': torch.LongTensor(self.tgt_tokens[index]),
			'node_tgt': self.node_tgt_tokens[index],
			'edge_tgt': self.edge_tgt_tokens[index],
		}	
		return item