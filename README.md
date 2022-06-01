# Setup
Add `src` directory to `PYTHONPATH` and install dependencies.
```
export PYTHONPATH='.'
pip install -r requirements.txt --use-feature=2020-resolver
```

# Experiments

After installing your dependencies, you can run the following scripts to reproduce our experiments:

### Preprocessing
Pulls COGS dataset, and preprocess sequential outputs into graph outputs under the `cogs/data` folder:
```
bash setup_cogs.sh
```
You can change the `TARGET_DIR` var in the above script to save the data in a different folder. 
Alternatively, for CFQ, you can run `bash setup_cfq.sh` which will perform the same and save data under `cfq/data`.

### COGS: Transformer Baseline (hyperparameters are set as default):
```
python src/train.py
```

### COGS: best strongly-supervised Transformer LAGr:
```
python src/train.py --from_config cogs/strongly_sup_hyperparams.yaml
```

### COGS/CFQ: Best weakly-supervised Transformer LAGr:
For COGS:
```
python src/train.py --from_config cogs/weakly_sup_hyperparams.yaml
```
For CFQ, run the following to train a model on the MCD1 split.
```
python src/cfq_train.py --from_config cfq/weakly_sup_hyperparams.yaml --split mcd1
```
