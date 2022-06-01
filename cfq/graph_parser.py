import argparse
import json
import os

import pandas as pd

def _maybe_adjust_label(node, q_type):
    if node == '?x0' and q_type == 'select_?x0':
        return 'select_?x0'
    return node


def parse(inp_text, out_text):
    """
    Takes input sequence and logical form, and identifies all nodes, edges, and their edge labels.
    """
    # splits logical form into triples
    out_text = out_text.split(' . ') 

    triples = []
    filters = []
    subj_only = []
    q_type = 'count(*)'

    for triple in out_text:
        if 'SELECT' in triple:
            if 'DISTINCT' in triple:
                q_type = 'select_?x0'
            continue

        if 'FILTER' in triple:
            filters.append(triple)
            continue

        # splits triple into entities
        triple = triple.split(' ')
        triple = [_maybe_adjust_label(ent.replace('^', ''), q_type) for ent in triple if ent not in ['a', '.']]
        # don't compress graph for nodes that only have a subject
        if len(triple) == 2:
            subj_only += [[(triple[0],), triple[1], (None,)]]
        else:
            triples += [triple]
        
    # reformat CFQ representation -> group by subject and predicate
    if triples:
        df = (
            pd.DataFrame(triples, columns=['subj', 'pred', 'obj'])
            .groupby(['pred', 'subj'])
            .agg({'obj': lambda x: tuple(x)})
            .reset_index()
            .groupby(['pred', 'obj'])
            .agg({'subj': lambda x: tuple(x)})
            .reset_index()
        )
    else:
        df = pd.DataFrame(triples, columns=['subj', 'pred', 'obj']) 
    if subj_only:
        df = pd.concat([df, pd.DataFrame(subj_only, columns=['subj', 'pred', 'obj'])])
    reformated_triples = df[['subj', 'pred', 'obj']].values.tolist() + filters

    nodes = []
    edges = []
    edge_labels = []
    for triple in reformated_triples:
        if 'FILTER' in triple:
            triple = triple.split(' ')
            triple[2] = _maybe_adjust_label(triple[2], q_type)
            triple[4] = _maybe_adjust_label(triple[4], q_type)
            edges.append([triple[2], triple[4]])
            edge_labels += ['FILTER']
        else:
            subjs, pred, objs = triple
            for sub in subjs:
                # change label if question is about x0
                if sub not in nodes:
                    nodes.append(sub)
                edges.append([sub, pred])
                edge_labels += ['agent']
            nodes.append(pred)

            for obj in objs:
                # change label if question is about x0
                if obj not in nodes and obj:
                    nodes.append(obj)
                if obj:
                    edges.append([pred, obj])
                    edge_labels += ['theme']
    
    results = {
        'original_inp': ' '.join(inp_text),
        'nodes': nodes,
        'edges': edges,
        'edge_labels': edge_labels,
        'outputs_triples': reformated_triples
    }
    return results


if __name__ == '__main__':

    args = argparse.ArgumentParser()

    args.add_argument("-data", "--data", default="train.tsv", type=str)
    args.add_argument("-split", "--split", default="mcd1", type=str)
    args.add_argument("-dir", "--dir", default="/cfq/sparql", type=str)
    args = args.parse_args()

    # create graph dir
    cfq_dir = os.path.join(args.dir, args.split)
    graph_path = os.path.join(cfq_dir, 'graph')

    with open(os.path.join(cfq_dir, args.data)) as f:
        df = pd.read_csv(f, sep='\t', header=None)
        df.columns = ['inp', 'out', '_']

    input_sequences = list(df['inp'].values)
    input_sequences = [sent.split(' ') for sent in input_sequences]
    output_sequences = list(df['out'].values)

    parsed_result = []

    for inp, out  in zip(input_sequences, output_sequences):
        results = parse(inp, out)
        parsed_result.append(results)

    dataset_name = 'dev' if args.data[:-4] == 'validation' else args.data[:-4]
    file_name = os.path.join(graph_path, dataset_name + '_parsed.json')
    with open(file_name, 'w') as f:
        json.dump(parsed_result, f)
