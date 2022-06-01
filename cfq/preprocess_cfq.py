import os
import tensorflow_datasets as tfds
import pandas as pd
import argparse


CFQ_SPLITS = ['random_split', 'mcd1', 'mcd2', 'mcd3']


def create_tsv(split_name, dir):
	ds = tfds.load('cfq/' + split_name)
	ds = tfds.as_numpy(ds)
	dir = os.path.join(dir, split_name)
	os.makedirs(os.path.join(dir, 'graph'), exist_ok=True)
	nodes, edges = [], []

	# saves split as tsv
	for split in ['train', 'test', 'validation']:
		df, new_nodes, new_edges = _clean_and_convert_to_df(ds, split)
		nodes += new_nodes
		edges += new_edges
		print(f"Saving {split_name}\'s {split} set as tsv ...")
		df.to_csv(
			os.path.join(dir, split + '.tsv'), sep="\t", header=False, index=False, encoding='utf-8')

	print(f'Saving nodes and edges under {dir} for {split_name} split ....')
	nodes = list(set(nodes))
	edges = list(set(edges))

	print(f'Extracted {len(nodes)} node labels.')
	print(f'Extracted {len(set(edges))} edges.')

	with open(os.path.join(dir,'graph/nodes.txt'), 'w') as f:
		for w in nodes:
			f.write(str(w) + '\n')

	with open(os.path.join(dir, 'graph/edges.txt'), 'w') as f:
		for w in edges:
			f.write(str(w) + '\n')


def _clean_and_convert_to_df(ds, split):
	"""
	Converts byte string to strings, and removes parts of query that aren't needed,
	and extracts nodes and edges for vocabulary.
	"""

	questions, queries = [],[]
	nodes, edges = {'vars':[], 'preds':[]}, ['agent', 'theme', 'FILTER']

	for example in ds[split]:
		query = example['query'].decode("utf-8") 
		questions.append(example['question'].decode("utf-8"))
		query = query.split('\n')

		# process question type
		question_type = query[0]
		if '?x0' in question_type:
			nodes['vars'].append('select_?x0')

		# process triple clauses
		query = query[1:-1]  # list of strings containing triples
		tmp_query = []
		for triple in query:
			tmp_query.append(triple)

			# Don't add new nodes from FILTER clauses. This should only induce a new edge across existing nodes
			if 'FILTER' not in triple:
				stripped = triple.split(' ')
				for entity in stripped:
		
					if 'ns:' in entity and 'ns:m' not in entity and 'ns:g' not in entity:
						if '^' in entity:
							# correct parsing error
							entity = entity[1:]
						nodes['preds'].append(entity)
					elif entity not in ['.', 'a']:
						nodes['vars'].append(entity)
		
		query = ' '.join([question_type + ' .'] + tmp_query)
		queries.append(query)
	
	nodes = nodes['vars'] + nodes['preds']

	placeholders = [None] * len(questions)
	df = pd.DataFrame(zip(questions, queries, placeholders), columns=['question', 'query', 'type'])
	return df, nodes, edges


if __name__ == "__main__":
	# creates data dir for split and creates tsv file
	args = argparse.ArgumentParser()
	args.add_argument("-dir", "--dir", default="/cfq/sparql", type=str)
	args = args.parse_args()

	for split in CFQ_SPLITS:
		create_tsv(split, args.dir)
