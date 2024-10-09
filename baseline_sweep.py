#!/usr/bin/env python
import json

import numpy as np
import torch
import torch.optim as optim

from model.Elem import Elem
from model.EmELpp import EmELpp
from model.Elbe import Elbe
from model.BoxEL import BoxEL
from model.TransBox_reg import TransBox
from utils.data_loader import DataLoader
import logging
from tqdm import trange
import wandb
from evaluate import compute_ranks, evaluate
from evaluate_complex_concept import evaluate_axiom_learning

from utils.utils import get_device
import sys

logging.basicConfig(level=logging.INFO)


def main():
    torch.manual_seed(42)
    np.random.seed(12)
    run()


def run():
    num_epochs = 5000
    wandb.init()

    dataset = wandb.config.dataset
    task = wandb.config.task
    embedding_dim = wandb.config.embedding_dim
    num_neg = wandb.config.num_neg if 'num_neg' in wandb.config else 1
    margin = wandb.config.margin if 'margin' in wandb.config else 0

    device = get_device()
    data_loader = DataLoader.from_task(task)
    train_data, classes, relations = data_loader.load_data(dataset)
    val_data = data_loader.load_val_data(dataset, classes)
    val_data['nf1'] = val_data['nf1'][:1000]
    print('Loaded data.')

    model_dict = {'elem': Elem, 'emelpp': EmELpp, 'elbe': Elbe, 'boxel': BoxEL,
                  'transbox': TransBox, 'transrolebox': TransRoleBox}
    model_name = wandb.config.model
    parameters = ''
    others = {}
    if model_name in {'transbox', 'transrolebox'}:
        neg_dist = wandb.config.neg_dist
        reg_factor = wandb.config.reg_factor
        if model_name == 'transbox':
            use_bias = wandb.config.use_bias
            others['use_bias'] = use_bias

            # todo: add number_components
            num_components = wandb.config.num_components
            others['num_components'] = num_components

            model = model_dict[model_name](device, classes, len(relations), embedding_dim, margin=margin,
                                           num_neg=num_neg, neg_dist=neg_dist, reg_factor=reg_factor, use_bias=use_bias,
                                           num_components=num_components)
        else:
            int_approx = wandb.config.int_approx
            others['int_approx'] = int_approx

            num_components = wandb.config.num_components
            others['num_components'] = num_components
            model = model_dict[model_name](device, classes, len(relations), embedding_dim, margin=margin,
                                           num_neg=num_neg, neg_dist=neg_dist, reg_factor=reg_factor,
                                           int_approx=int_approx, num_components=num_components)

        parameters += (f'lr:{wandb.config.lr}, embedding_dim:{embedding_dim}, num_neg:{num_neg}, '
                       f'neg_dist:{neg_dist}, reg_factor:{reg_factor},  margin:{margin}')

    elif model_name == 'boxel':
        model = model_dict[model_name](device, classes, len(relations), embedding_dim)
        parameters += f'lr:{wandb.config.lr}, embedding_dim:{embedding_dim}'
    else:
        model = model_dict[model_name](device, classes, len(relations), embedding_dim, margin=margin)
        parameters += f'lr:{wandb.config.lr}, embedding_dim:{embedding_dim}, margin:{margin}'

    out_folder = f'data/{dataset}/{task}/{model.name}'

    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)
    scheduler = None
    model = model.to(device)

    if not model.negative_sampling and task != 'old':
        sample_negatives(train_data, num_neg)

    train(model, train_data, val_data, len(classes), optimizer, scheduler, out_folder, num_epochs=num_epochs,
          val_freq=100)

    print('Computing test scores...')
    scores = evaluate(dataset, task, model.name, embedding_size=model.embedding_dim,
                      parameters=parameters, best=True, split='test', others=others)
    combined_scores = scores[-1]
    surrogate = np.median(combined_scores.ranks) - combined_scores.top100 / len(combined_scores) - \
                0.1 * combined_scores.top10 / len(combined_scores)
    wandb.log({'surrogate': surrogate})

    AL_scores = evaluate_axiom_learning(dataset, 'prediction', model_name=model_name, embedding_size=model.embedding_dim, best=True)

    # print("scores based on volume:")
    # if model_name != 'boxel':
    #     _ = evaluate_volume(dataset, task, model.name, embedding_size=model.embedding_dim,
    #                         parameters=parameters, best=True, split='test', others=others)

    wandb.finish()
    return scores, AL_scores


def train(model, data, val_data, num_classes, optimizer, scheduler, out_folder, num_epochs=2000, val_freq=100):
    model.train()
    wandb.watch(model)

    best_top10 = 0
    best_top100 = 0
    best_median = sys.maxsize
    best_mean = sys.maxsize
    best_epoch = 0

    try:
        for epoch in trange(num_epochs):
            try:
                loss = model(data, epoch)
            except:
                loss = model(data)

            if epoch % val_freq == 0 and val_data is not None:
                ranking = compute_ranks(model.to_loaded_model(), val_data, num_classes, 'nf1', model.device)
                wandb.log({'top10': ranking.top10 / len(ranking), 'top100': ranking.top100 / len(ranking),
                           'mean_rank': np.mean(ranking.ranks), 'median_rank': np.median(ranking.ranks)}, commit=False)
                # if ranking.top100 >= best_top100:
                if np.median(ranking.ranks) <= best_median:
                    # if np.mean(ranking.ranks) <= best_mean:
                    best_top10 = ranking.top10
                    best_top100 = ranking.top100
                    best_median = np.median(ranking.ranks)
                    best_mean = np.mean(ranking.ranks)
                    best_epoch = epoch
                    model.save(out_folder, best=True)
            wandb.log({'loss': loss})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    except KeyboardInterrupt:
        print('Interrupted. Stopping training...')

    print(f'Best epoch: {best_epoch}')
    model.save(out_folder)


def sample_negatives(data, num_neg):
    for i in range(num_neg):
        nf3 = data['nf3']
        randoms = np.random.choice(data['class_ids'], size=(nf3.shape[0], 2))
        randoms = torch.from_numpy(randoms)
        new_tails = torch.cat([nf3[:, [0, 1]], randoms[:, 0].reshape(-1, 1)], dim=1)
        new_heads = torch.cat([randoms[:, 1].reshape(-1, 1), nf3[:, [1, 2]]], dim=1)
        new_neg = torch.cat([new_tails, new_heads], dim=0)
        data[f'nf3_neg{i}'] = new_neg


if __name__ == '__main__':
    main()
