import json
import os

import numpy as np
import pandas as pd
import torch
import torchtext
from collections import Counter, OrderedDict
from torch.nn.utils.rnn import pad_sequence as pad
from transformers import AutoTokenizer


VARIABLES = ['?x0', '?x1', '?x2', '?x3', '?x4', '?x5', 'select_?x0']

NONPREDICATES = VARIABLES + [
    'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', \
    'ns:m.02zsn', 'ns:m.0345h', 'ns:m.03_3d', 'ns:m.03rjj', 'ns:m.059j2', \
    'ns:m.05zppz', 'ns:m.06mkj', 'ns:m.07ssc', 'ns:m.09c7w0', 'ns:m.0b90_r', \
        'ns:m.0d05w3', 'ns:m.0d060g', 'ns:m.0d0vqn', 'ns:m.0f8l9c'
]

GRAPH_PAD_TOKEN = 1
NULL_TOKEN = 0


class Preprocessor():

    def __init__(self, data_dir='data/graph/', graph_layers=1):
        self.data_dir = data_dir
        self.node_vocab, self.edge_vocab = self.tokenize()
        self.graph_layers = graph_layers
        self.root = '/'.join(self.data_dir.split('/')[:-1])
        
        assert self.word_to_token('<pad>') == GRAPH_PAD_TOKEN, f'<pad> token needs to be mapped to 1, but got {GRAPH_PAD_TOKEN}'
        assert self.word_to_token('null') == NULL_TOKEN, f'Null token needs to be mapped to 0, but got {NULL_TOKEN}'
        assert self.word_to_token('<pad>', 'edges') == GRAPH_PAD_TOKEN, f'<pad> token needs to be mapped to 1, but got {GRAPH_PAD_TOKEN}'
        assert self.word_to_token('null', 'edges') == NULL_TOKEN, f'Null token needs to be mapped to 0, but got {NULL_TOKEN}'

    def tokenize(self):
        with open(os.path.join(self.data_dir, 'nodes.txt'), 'r') as f:
            nodes = [line.rstrip() for line in f]

        # vocab will use null token:0, <pad>_token:1
        node_vocab = torchtext.vocab.Vocab(Counter(nodes), specials=['null', '<pad>'])

        with open(os.path.join(self.data_dir, 'edges.txt'), 'r') as f:
            edges = [line.rstrip() for line in f]

        # vocab will use null token:0, <pad>_token:1
        edge_vocab = torchtext.vocab.Vocab(Counter(edges), specials=['null', '<pad>'])

        return node_vocab, edge_vocab

    def word_to_token(self, word, vocab='nodes'):
        if vocab == 'nodes':
            return self.node_vocab.stoi[word]
        return self.edge_vocab.stoi[word]

    def token_to_word(self, token, vocab='nodes'):
        if vocab == 'nodes':
            return self.node_vocab.itos[token]
        return self.edge_vocab.itos[token]

    def scores_to_graph(self, node_preds, edge_preds, node_targets, edge_targets):
        """ returns the batch of nodes, edges and edge labels where all three include null nodes and null edges, but exclude pad predictions."""

        assert node_preds.shape[0] == edge_preds.shape[0], \
            f'Expected batch first, but got {node_preds.shape[0]}, {edge_preds.shape[0]}'

        assert node_preds.shape == node_targets.shape

        node_sets = []
        for sequence, target_seq in zip(node_preds, node_targets):  # BS, seq_len
            nodes = {}
            for idx, (token, target_token) in enumerate(zip(sequence, target_seq)):  # seq_len
                token = token.item()
                node = self.token_to_word(token, vocab='nodes')

                # if the target is a pad token, or if the prediction is pad, ignore it
                if node == '<pad>' or target_token == GRAPH_PAD_TOKEN:
                    continue
                else:
                    nodes[f'x_{idx}'] = node
            node_sets.append(nodes)

        edge_sets = []
        edge_label_sets = []
        for sequence, target_seq in zip(edge_preds, edge_targets):  # BS, seq_len, seq_len
            edges = []
            edge_labels = []

            for token_i in range(sequence.shape[0]):
                for token_j in range(sequence.shape[0]):

                    predicted_token = sequence[token_i, token_j]
                    edge_label = self.token_to_word(predicted_token, vocab='edges')
                    true_token = target_seq[token_i, token_j]
                    true_label = self.token_to_word(true_token, vocab='edges')

                    if edge_label == '<pad>' or true_label == '<pad>':
                        continue
                    else:
                        edge = [f'x_{token_i}', f'x_{token_j}']
                        edges.append(edge)
                        edge_labels.append(edge_label)

            edge_sets.append(edges)
            edge_label_sets.append(edge_labels)

        return node_sets, edge_sets, edge_label_sets

    def graph_to_lambda(self, node_sets, edge_sets, edge_labels):
        """
        Returns predicted serialized lambda string or NA if graph could not be serialized.
        """
        lambdas = []
        for nodes, edges, labels in zip(node_sets, edge_sets, edge_labels):
            # removes null nodes
            length = len(nodes)
            nodes = {k: v for k, v in nodes.items() if nodes[k] != 'null'}
            try:
                # removes null edges
                edges, labels = zip(*[(edge, label) for edge, label in zip(edges, labels) if label != 'null'])
                lmbd = self._graph_to_lambda(nodes, edges, labels, length)
            except:
                # skip example if there are no non-null edges
                lmbd = 'NA'

            lambdas.append(lmbd)
        return lambdas

    def _graph_to_lambda(self, nodes, edges, edge_labels):
        """
        Turns a graph into a set of clauses serialized in the same way as COGS does.
        """
        # containing * word clauses - always goes to the beginning
        new = {}
        for k,v in nodes.items():
            if k == '*':
                continue
            new[k] = v
            
        for idx, items in enumerate(zip(edges, edge_labels)):
            (e1, e2), label = items
            if label == 'article' and nodes.get(e1) == '*':
                new[e2] = f'* {nodes[e2]}'
        
        nodes=new
        definite_clauses = OrderedDict()
        

        # containing rest of the clauses - ordered primarily by arg1 and by arg2, if available
        event_clauses = OrderedDict({f'x_{idx}': {} for idx in range(40)})

        for idx, items in enumerate(zip(edges, edge_labels)):
            edge, label = items
            # e.g. x _ 1, x _ 3
            e1, e2 = edge
            
            if label == 'article':
                continue

            # retrieves node labels, except if they are null nodes
            try:
                node_label1 = nodes[e1]
                node_label2 = nodes[e2]
            except:
                continue

            # reformat to x_1, x_3
            e1 = e1.replace(' ', '')
            e2 = e2.replace(' ', '')

            # if edge contains named entity, only produce one clause
            if node_label1[0].isupper() and node_label2[0].isupper():
                print('Edge connects two named entities. Unexpected behavior.')
                continue
            if node_label1[0].isupper():
                named_entity = node_label1
                event_clauses[e2][e1] = node_label2 + ' . ' + label + ' (' + e2 + ', ' + named_entity + ') '
            elif node_label2[0].isupper():
                named_entity = node_label2
                event_clauses[e1][e2] = node_label1 + ' . ' + label + ' (' + e1 + ', ' + named_entity + ') '
            # if not named entity
            else:
                ## if * word appears, modify its name in the arg
                if '*' in node_label1 and '*' in node_label2:
                    # print('Two definite articles connected. Unexpected behavior')
                    # import ipdb;ipdb.set_trace()

                    definite_clauses[e1] = node_label1 + ' (' + e1 + ')'
                    definite_clauses[e2] = node_label2 + ' (' + e2 + ')'
                    node_label1 = node_label1[2:]
                    #             node_label2 = node_label2[2:]
                    event_clauses[e1][e2] = node_label1 + ' . ' + label + ' (' + e1 + ', ' + e2 + ')'

                elif '*' in node_label1:
                    definite_clauses[e1] = node_label1 + ' (' + e1 + ')'
                    node_label1 = node_label1[2:]
                    event_clauses[e1][e2] = node_label1 + ' . ' + label + ' (' + e1 + ', ' + e2 + ')'
                    event_clauses[e2] = {'_': node_label2 + ' (' + e2 + ')'}

                elif '*' in node_label2:
                    definite_clauses[e2] = node_label2 + ' (' + e2 + ')'
                    node_label2 = node_label2[2:]
                    event_clauses[e1][e2] = node_label1 + ' . ' + label + ' (' + e1 + ', ' + e2 + ')'

                else:
                    event_clauses[e1][e2] = node_label1 + ' . ' + label + ' (' + e1 + ', ' + e2 + ')'
                    event_clauses[e2] = {'_': node_label2 + ' (' + e2 + ')'}

        # produces serialized output
        final = ''
        definite_clause_keys = sorted([int(k[2:]) for k in definite_clauses.keys()])
        for key in definite_clause_keys:
            final += definite_clauses['x_' + str(key)] + ' ; '

        event_clause_keys = sorted([int(k[2:]) for k, v in event_clauses.items() if v != {}])

        for key in event_clause_keys:
            subkeys = sorted([int(k[2:]) if k != '_' else -1 for k in event_clauses['x_' + str(key)].keys()])

            for clause_key in subkeys:
                clause_key = 'x_' + str(clause_key) if clause_key != -1 else '_'

                if clause_key == '_':
                    if len(event_clauses['x_' + str(key)]) > 1 and \
                            not np.any(['nmod' in val for val in event_clauses['x_' + str(key)].values()]):
                        continue

                final += event_clauses['x_' + str(key)][clause_key] + ' AND '

    #     removes AND from the end
        try:
            final = final[:-5]
            if final[-1] == ' ':
                final = final[:-1]
        except:
            return 'NA'

        return final.replace('x_', ' x _ ').replace(')', ' )').replace('  ', ' ').replace(',', ' ,')


def _cogs_build_graph(graph, preprocessor, length, n_special_tokens):
    node_tgt_tensor = _cogs_nodes_to_node_seq(graph['nodes'], length, preprocessor, n_special_tokens)
    edge_tgt_tensor = _cogs_edges_to_edge_seq(graph['edges'], graph['edge_labels'], length, preprocessor, n_special_tokens)
    return node_tgt_tensor, edge_tgt_tensor


def _cfq_build_graph(graph, preprocessor, length, n_special_tokens):
    node_tgt_tensor = _cfq_nodes_to_node_seq(graph['nodes'], length, preprocessor, n_special_tokens)
    edge_tgt_tensor = _cfq_edges_to_edge_seq(graph['edges'], graph['edge_labels'], 
        graph['nodes'], graph['outputs_triples'], length, preprocessor, n_special_tokens)
    return node_tgt_tensor, edge_tgt_tensor


def _cogs_nodes_to_node_seq(nodes, input_length, preprocessor, n_special_tokens=0):
    """
    Takes nodes dict and turns it into a sequence of node tags for each input token.
    nodes = {'x_1': node1, 'x_2': node2}
    input_length: length of ['some', input, sentence]
    preprocessor = object containing graph vocabulary
    returns
            - node token sequence (torch.LongTensor) ( corresponding to [null, node1, node2] )
    """
    node_sequence = torch.LongTensor([NULL_TOKEN for _ in range(input_length)])

    for node, node_label in nodes.items():
        node_sequence[int(node[2:])] = preprocessor.word_to_token(node_label)

    # Adds padding for SOS and EOS tokens
    pad = torch.tensor([GRAPH_PAD_TOKEN] * preprocessor.graph_layers)
    return torch.cat([pad, node_sequence, pad], 0)

def _cogs_edges_to_edge_seq(edges, edge_labels, N, preprocessor, n_special_tokens=0):
    """
    Takes edges and turns it into a sequence of set of edge tags for each input token.
    nodes = {'x_0': word0, 'x_1': word1, 'x_2': Levi}
    edges = [['x_1', 'x_0'], ['x_2', 'x_1']]
    N: length of ['some', input, sentence]
    edge_labols = [theme, agent]
    preprocessor = object containing graph vocabulary

    returns
        torch tensor [[1,1,0,0], [1,1,0,0], [0,0,1,1], [0,0,1,0]]
    """

    # default edge type: null (no edge)
    final_edges = torch.LongTensor([NULL_TOKEN] * N * N).view(N, N)

    for edge, label in zip(edges, edge_labels):
        e1, e2 = [int(e[2:]) for e in edge]
        label_token = preprocessor.word_to_token(label, vocab='edges')
        final_edges[e1, e2] = label_token
    
    # add padding for SOS and EOS tokens
    N = N + n_special_tokens * preprocessor.graph_layers
    padding_tokens = torch.LongTensor([GRAPH_PAD_TOKEN] * N * N).view(N, N)
    padding_tokens[1:-1, 1:-1] = final_edges
    
    return padding_tokens


def _cfq_nodes_to_node_seq(nodes, num_nodes, preprocessor, n_special_tokens=0):
    padding = num_nodes - len(nodes)
    nodes = nodes + ['null'] * padding
    node_sequence = torch.LongTensor([NULL_TOKEN for _ in range(num_nodes)])

    for pos in range(num_nodes):
        node_sequence[pos] = preprocessor.word_to_token(nodes[pos])

    # Adds padding for SOS and EOS tokens
    pad = torch.tensor([GRAPH_PAD_TOKEN] * preprocessor.graph_layers)
    return torch.cat([pad, node_sequence, pad], 0)


def _cfq_edges_to_edge_seq(edges, edge_labels, nodes, output_triples, N, preprocessor, n_special_tokens=0):
        
    assert len(nodes) <= N, f"Too many nodes ({len(nodes)}) for the number of node predictions ({N})"

    # default edge type: null (no edge)
    final_edges = torch.LongTensor([preprocessor.word_to_token('null', vocab='edges')] * N * N).view(N, N)
    node_counter = Counter(nodes)
    repeating_nodes = [node for node, freq in node_counter.items() if freq > 1]
    relevant_edges = []
    for tri in output_triples:
        for node in repeating_nodes:
            if tri[1] == node:
                relevant_edges.append(tri)

    for edge, label in zip(edges, edge_labels):
        src, dest = edge  # edge contains str node labels
        pred_id = None

        if src in repeating_nodes and dest in repeating_nodes:
            raise(f'This should never happen, but got src: {src}, and obj: {obj}')

        # handle repeated node labels
        if src in repeating_nodes:
            objects = [e[-1] for e in relevant_edges if e[1] == src]
            for idx, objs in enumerate(objects):
                for obj in objs:
                    if dest == obj:
                        pred_id = idx
                        break
                if pred_id:
                    break

            pointer = 0
            for idx, n in enumerate(nodes):
                if n == src:
                    if pointer == pred_id:
                        src_id = idx
                        break
                    else:
                        pointer += 1
        else:
            src_id = nodes.index(src) 

        if dest in repeating_nodes:
            subjects = [e[0] for e in relevant_edges if e[1] == dest]
            for idx, subjs in enumerate(subjects):
                for sub in subjs:
                    if src == sub:
                        pred_id = idx
                        break
                if pred_id:
                    break

            pointer = 0
            for idx, n in enumerate(nodes):
                if n == dest:
                    if pointer == pred_id:
                        dest_id = idx
                        break
                    else:
                        pointer += 1
        else:
            dest_id = nodes.index(dest)

        label_token = preprocessor.word_to_token(label, vocab='edges')
        try:
            final_edges[src_id, dest_id] = label_token
        except:
            raise Exception('Unexpected behavior.')

    # Adds padding for src sequence's SOS and EOS
    N = N + n_special_tokens * preprocessor.graph_layers
    padding_tokens = torch.LongTensor([GRAPH_PAD_TOKEN] * N * N).view(N, N)
    skip = preprocessor.graph_layers
    padding_tokens[skip:-skip, skip:-skip] = final_edges

    return padding_tokens