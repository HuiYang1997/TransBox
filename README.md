# TransBox: EL++-closed ontology embeddings

This repository built upon implementation of the paper [Dual Box Embeddings for the Description Logic EL++](https://arxiv.org/abs/2301.11118).

## Requirements

This current implementation is under PyTorch 2.3.1, with python package:
`mowl==0.3.0`, `joblib==1.4.2`, `numpy==1.26.4`, `tqdm==4.64.0`, `wandb==0.13.9`.

## Data

To obtain the data, unzip `data.zip`:

```sh
unzip data.zip
```

The algorithm of generating the test data of complex axioms can be found in: `generateComplexAxioms/create_complex_axioms_by_forgetting.py`.

## Training and testing
For runing the training and testing of the model, you can use the following command:

```sh
python run_many_seeds.py
```

The hyperparameters can be modified in the `configures_ours.json` file.