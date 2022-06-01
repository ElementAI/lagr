import argparse
import json
import os
import re

import pandas as pd
from src.utils import DATASET_FILENAMES


def two_place_predicates_without_nmod(s):
    """
    Takes s and looks for the following pattern:
      word . word (arg1, arg2) 
    Excludes word . word . word (...) patterns.

    Returns: 
      [word, word, arg1, arg2]
    """
    pattern = re.findall(r"(?:^|\W)(?<!(\.\ ))(\w+)\ \.\ (\w+)\ \((.*?)\)", s)
    result = []
    for p in pattern: 
        result.append(p[1:])
    return result


def two_place_predicates_with_nmod(s):
    """
    Takes s and looks for the following pattern:
      word . word (arg1, arg2) 
    Returns: 
      [word, word, arg1, arg2]
    """
    pattern = re.findall(r"(\w+)\ \.\ (\w+)\ \.\ (\w+).*? \((.*?)\)", s)
    result = []
    for p in pattern:
        res = (p[0], p[1] + ' . ' + p[2], p[3])
        assert len(p) == 4, f"Should be nmod clause, but got {p}"
        result.append(res)
    return result


def one_place_definite_predicate(s):
    """
    Takes s and looks for the following pattern:
      * cake (x _ 3)
    Returns:
      [('* ', 'cake', 'x _ 3')]
    """
    return re.findall(r"(?:^|\W)(?<=(\*\ ))(\w+) \((.*?)\)", s)


def one_place_indefinite_predicate(s):
    """
    Takes s and looks for the following pattern:
      word (x _ 1) 
    Returns:
      ['', '', 'word', 'x _ 1']
    """
    return re.findall(r"(?:^|\W)(?<!(\.\ )|(\*\ ))(\w+) \((.*?)\)", s) 


def extract_graph_components(pattern, nodes, edges, edge_labels, inp_text):
    for p in pattern:
        assert len(p) == 3, f'should be 3 components to account for 2-arg predicates, but got {len(p)} for {p}'

        node_label = p[0]
        edge_type = p[1]
        args = p[2].replace(' ','').split(',')
        arg1 = args[0]
        arg2 = args[1]

        if len(args) != 2:
            raise Exception(f"You likely didn\'t exclude primite types. Your out_text does not follow the regular parse patterns.")

        if not nodes.get(arg1):
            nodes[arg1] = node_label

        # if not x_id arg, but named entity
        if 'x_' not in arg2:
            assert arg2[0].isupper() == True, f"Named entity doesnt start with capital, {arg2}."

            matched_indices = [idx for idx, w in enumerate(inp_text) if w == arg2]
            assert len(matched_indices) == 1, f'{arg2} has multiple matches in {inp_text}.'
            idx = matched_indices[0]
            node = arg2
            arg2 = 'x_' + str(idx)
            nodes[arg2] = node

        edges.append((arg1, arg2))
        edge_labels.append(edge_type)

    return nodes, edges, edge_labels


def primitive_res(inp_text):
    return {
        'original_inp': ' '.join(inp_text),
        'nodes': {'x_0': inp_text[0]},
        'edges': [[]],
        'edge_labels': []
    }


def parse(inp_text, out_text, type):
    """
    Takes lambda expression and parses it into a set of nodes, edges, and edge_labels as follows:
    word.other(x_idx1, x_idx2) -> word is x_idx1 node, with word node label
                               -> x_idx2 will have node label from input_seq[idx2]
                               -> other is edge label between x_idx1 and x_idx2
    word.other(x_idx1, word2)  -> word2 is node, with node label from a "searching the index of word" in the input.
    word(x_idx1)               -> word is x_idx1 label
    * word(x_idx1)             -> '* word' is x_idx1 label
    
    The above pattern break if:
    - word2 isn't exactly mentioned in the input sequence.
    """
    if type == 'primitive':
        return primitive_res(inp_text)

    nodes = {}
    edges = []
    edge_labels = []

    # inp: word . word (arg1, arg2) out: [word, word, arg1, arg2]
    # e.g. [(forward, theme, arg1, arg2), (forward, agent, arg1, arg2)]
    pattern = two_place_predicates_without_nmod(out_text)
    nodes, edges, edge_labels = extract_graph_components(pattern, nodes, edges, edge_labels, inp_text)

    if 'nmod' in out_text:
        pattern = two_place_predicates_with_nmod(out_text)
        nodes, edges, edge_labels = extract_graph_components(pattern, nodes, edges, edge_labels, inp_text)


    # finds * cake (x _ 3) -> [('* ', 'cake', 'x _ 3')]
    pattern = one_place_definite_predicate(out_text)

    for p in pattern:
        assert p[0] == '* ', f'Didn\'t extract * in {p}'
        assert len(p) == 3, 'Missing patterns. Likely to be the wrong pattern.'
        node = p[1]
        node_id = p[2].replace(' ','')
        nodes[node_id] = node
        # save star as a separate node, indexed one before the node label
        star_position = f'x_{str(int(node_id[2:]) - 1)}'
        nodes[star_position] = '*'
        edges.append((star_position, node_id))
        edge_labels.append(('article'))

    # word (x _ 1) not preceded by . => ['', '', 'word', 'x _ 1']
    pattern3 = one_place_indefinite_predicate(out_text) 
    pattern3 = [e[2:] for e in pattern3]

    for p in pattern3:
        assert len(p) == 2, f'Should only have 1 arg, but extracted {p}.'
        node = p[0]
        node_id = p[1].replace(' ', '')
        nodes[node_id] = node

    results = {
        'original_inp': ' '.join(inp_text),
        'nodes': nodes,
        'edges': edges,
        'edge_labels': edge_labels
    }
    return results


def show_parsed(original, parsed, N=1):
    for idx, orig in enumerate(original[:N]):
        print('original input:\n', orig[0])
        print('original output:\n', orig[1])
        print('Parsed: \n')
        print('nodes       ', parsed[idx]['nodes'])
        print('edges       ', parsed[idx]['edges'])
        print('edge_labels ', parsed[idx]['edge_labels'])
        print('\n########')


def build_vocabularies(dir):
    nodes_vocab = set()
    data = json.load(open(f'{dir}/graph/train_parsed.json', 'r'))
    for d in data:
        nodes = d['nodes'].values()
        for n in nodes:
            nodes_vocab.add(n)
    data = json.load(open(f'{dir}/graph/test_parsed.json', 'r'))
    for d in data:
        nodes = d['nodes'].values()
        for n in nodes:
            nodes_vocab.add(n)
    data = json.load(open(f'{dir}/graph/dev_parsed.json', 'r'))
    for d in data:
        nodes = d['nodes'].values()
        for n in nodes:
            nodes_vocab.add(n)
    data = json.load(open(f'{dir}/graph/gen_parsed.json', 'r'))
    for d in data:
        nodes = d['nodes'].values()
        for n in nodes:
            nodes_vocab.add(n)

    nodes_vocab = list(nodes_vocab)

    edges_vocab = set()

    data = json.load(open(f'{dir}/graph/train_parsed.json', 'r'))
    for d in data:
        edges = d['edge_labels']
        for e in edges:
            edges_vocab.add(e)

    data = json.load(open(f'{dir}/graph/test_parsed.json', 'r'))
    for d in data:
        edges = d['edge_labels']
        for e in edges:
            edges_vocab.add(e)

    data = json.load(open(f'{dir}/graph/dev_parsed.json', 'r'))
    for d in data:
        edges = d['edge_labels']
        for e in edges:
            edges_vocab.add(e)

    data = json.load(open(f'{dir}/graph/gen_parsed.json', 'r'))
    for d in data:
        edges = d['edge_labels']
        for e in edges:
            edges_vocab.add(e)

    edges_vocab = list(edges_vocab)

    with open(f'{dir}/graph/edges.txt', 'w') as f:
        for w in edges_vocab:
            f.write(str(w) + '\n')

    with open(f'{dir}/graph/nodes.txt', 'w') as f:
        for w in nodes_vocab:
            f.write(str(w) + '\n')


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("-show", "--show", type=bool)
    args.add_argument("-data_dir", "--data_dir", type=str)
    args = args.parse_args()
    data_dir = args.data_dir

    # check if cogs data dir exists
    data_path = os.path.join(data_dir, 'lambda')
    if not os.path.exists(data_path):
        raise Exception(f'COGS data under {data_path} doesn\'t exist.')

    # create graph dir
    graph_path= os.path.join(data_dir, 'graph')
    if not os.path.exists(graph_path):
        print('Creating dir for graph data...')
        os.makedirs(graph_path)

    # create graph data 
    for file in DATASET_FILENAMES:
        file_name = file + '.tsv'
 
        with open(os.path.join(data_path, file_name)) as f:
            df = pd.read_csv(f, sep='\t', header=None)
            df.columns = ['inp', 'out', 'type']

        # assert df[df.type == 'primitive'].size == 0

        input_sequences = list(df['inp'].values)
        types = list(df['type'].values)
        input_sequences = [sent.split(' ') for sent in input_sequences]
        output_sequences = list(df['out'].values)

        parsed_result = []

        for inp, out, type in zip(input_sequences, output_sequences, types):
            results = parse(inp, out, type)
            parsed_result.append(results)

        if args.show:
            show_parsed(list(zip(input_sequences, output_sequences)), parsed_result, N=5)

        # savings parsed graphs
        file_name = os.path.join(data_dir, f'graph/{file}_parsed.json')
        with open(file_name, 'w') as f:
            json.dump(parsed_result, f)

    # Build vocabularies from all datafiles
    build_vocabularies(data_dir)