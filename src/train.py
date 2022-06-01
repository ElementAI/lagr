import numpy as np
import os
import sys
import torch
import argparse
import wandb

from src.model import Parser
from src.utils import (
	set_random_seed, setup_data, CustomDataset, 
	pad_tensors, parse_args, 
	_maybe_fetch_ckpt, _ckpt_callback, _create_name, make_collator
)
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, SubsetRandomSampler

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/cogs', type=str, help='Parent path for dataset.')
parser.add_argument('--split', default='', type=str, help='CFQ data split', choices=['random_split', 'mcd1', 'mcd2', 'mcd3', ''])
parser.add_argument('--data', default='cogs', type=str, help='Dataset name', choices=['cogs','cfq'])
parser.add_argument('--from_config', type=str, default='')
parser.add_argument('--ckpt', type=str, default='')
parser.add_argument('--model_type', type=str, default="transformer_baseline", choices=['transformer_lagr', 'transformer_baseline'])
parser.add_argument("--eval_every", type=int, default=20)
parser.add_argument("--out_file", type=str, default='logging/predictions.jsonl')

parser.add_argument('--optimizer',type=str,default="Adam")
parser.add_argument('--scheduler',type=str,default="linear_warmup",choices=["linear_warmup"])
parser.add_argument("--lr", default=1e-4)
parser.add_argument("--weight_decay", default=0.0)
parser.add_argument('--epochs',type=int, default=200)
parser.add_argument('--batch_size',type=int, default=128)
parser.add_argument('--gen_batch_size',type=int, default=128)
parser.add_argument('--num_warmup_steps',type=int, default=0)
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-08)
parser.add_argument('--accum_grad',type=int, default=1)
parser.add_argument('--dropout',type=float, default=0.1)
parser.add_argument('--dim',type=int, default=512)

# default from csordas:
parser.add_argument("-transformer.ff_multiplier", default=1.0)
parser.add_argument("-transformer.encoder_n_layers", default=2)
parser.add_argument("-transformer.decoder_n_layers", default=2)
parser.add_argument("-transformer.tied_embedding", default=True)
parser.add_argument("-transformer.n_heads", default=4)

# LAGr args
parser.add_argument('--k', '-k', type=int, default=1)
parser.add_argument('--noise', '-noise', type=float, default=0.001)
parser.add_argument('--weak_supervision', '-weak_supervision', action='store_true', default=False)
parser.add_argument('--supervised', '-supervised', dest='weak_supervision', action='store_false')
parser.add_argument('--n_graph_layers',type=int, default=1)
parser.add_argument('--n_node_labels',type=int, default=646)  # cogs
parser.add_argument('--n_edge_labels',type=int, default=11)   # cogs
parser.add_argument('--share_encoder', '-share_encoder', action='store_true', default=True)
parser.add_argument('--separate_encoders', '-separate_encoders', dest='share_encoder', action='store_false')

parser.add_argument('--precision',type=int,default=16,choices=[16,32])
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--interactive', action='store_true', default=False)


cl_args = parse_args(parser)
print(cl_args)


def train():

	datasets, vocabs, preprocessor = setup_data(cl_args)
	src_vocab, tgt_vocab = vocabs
	cl_args.vocab_size = len(src_vocab)   # includes PAD
	cl_args.tgt_vocab_size = len(tgt_vocab)
	cl_args.src_vocab = src_vocab
	cl_args.tgt_vocab = tgt_vocab
	cl_args.preprocessor = preprocessor
	
	training_set = CustomDataset(datasets['train'])
	val_set = CustomDataset(datasets['gen_dev'])
	gen_set = CustomDataset(datasets['gen'])
	test_set = CustomDataset(datasets['test'])

	train_collator = make_collator()
	val_collator = make_collator()
	test_collator = make_collator()
	gen_collator = make_collator()

	train_params = {'batch_size': cl_args.batch_size, 'shuffle': True, 'drop_last':True, 'num_workers':8}
	val_params = {'batch_size': cl_args.batch_size, 'shuffle': False, 'num_workers': 8}
	test_params = {'batch_size': cl_args.gen_batch_size, 'shuffle': False, 'num_workers': 8}
	gen_params = {'batch_size': cl_args.gen_batch_size, 'shuffle': False, 'num_workers':8}	
	sample_ids = torch.randint(len(training_set), (int(len(training_set) * 0.1),))
	train_val_params = {'batch_size': cl_args.batch_size, 'shuffle': False, 'num_workers': 8, 'sampler': SubsetRandomSampler(sample_ids)}
				
	train_loader = DataLoader(training_set, **train_params,collate_fn=train_collator)
	val_loader = DataLoader(val_set, **val_params,collate_fn=val_collator)
	test_loader = DataLoader(test_set, **test_params,collate_fn=test_collator)
	gen_loader = DataLoader(gen_set, **gen_params,collate_fn=gen_collator)
	train_val_loader = DataLoader(training_set, **train_val_params, collate_fn=val_collator)

	optimizer_args = {'lr': cl_args.lr, 'betas': (cl_args.beta1,cl_args.beta2), 'eps': cl_args.eps, 'weight_decay': cl_args.weight_decay}
	num_training_steps = len(training_set) / (cl_args.batch_size * cl_args.accum_grad) * cl_args.epochs
	cl_args.expected_cache_size = len(training_set)
	print(f'Size of trainset: {len(training_set)} - Num train steps: {num_training_steps}')

	scheduler_args = {'num_warmup_steps':cl_args.num_warmup_steps, 'num_training_steps':num_training_steps}
	cl_args.optimizer_args = optimizer_args
	cl_args.scheduler_args = scheduler_args

	exp_name, run_name = _create_name(cl_args)
	cl_args.ckpt, possible_run_id = _maybe_fetch_ckpt(cl_args, exp_name, run_name)   # if resumed, try fetching last best ckpt

	# No wandb in interactive mode
	if not cl_args.interactive:
		logger = WandbLogger(name=run_name, group=exp_name, project="lagr", entity='dorajam', config=cl_args, resume="allow", id=os.environ.get("EAI_JOB_ID", ''))
	else:
		logger = None

	if cl_args.ckpt:
		model = Parser.load_from_checkpoint(cl_args.ckpt)
	else:
		model = Parser(cl_args)

	checkpoint_callback = _ckpt_callback(cl_args, exp_name, run_name, possible_run_id)
	trainer = pl.Trainer(
			max_epochs=cl_args.epochs,
			gradient_clip_val=cl_args.max_grad_norm,
			# deterministic=True,      # reproducibility
			gpus=1,                  # run on gpu
			precision=cl_args.precision,
			check_val_every_n_epoch=cl_args.eval_every,
			num_sanity_val_steps=0,
			logger=logger,      # log on wandb
			callbacks=[checkpoint_callback]
	)
	trainer.fit(model, train_loader, val_dataloaders=[val_loader, train_val_loader], ckpt_path=cl_args.ckpt)   # resume if ckpt is provided
	test_result = trainer.test(model, test_dataloaders=[train_loader, test_loader, gen_loader], ckpt_path='best')
	print(test_result)
	
	# exit code for random restart
	sys.exit(0)


if __name__=="__main__":
	set_random_seed(cl_args.seed)
	train()
