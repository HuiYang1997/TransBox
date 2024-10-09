from collections import Counter

import logging
import numpy as np
import torch
import torch.nn.functional as F
import datetime
import math
from tqdm import trange

from ranking_result import RankingResult
from utils.utils import get_device
from utils.data_loader import DataLoader
from model.loaded_models import LoadedModel, BoxELLoadedModel, TransBoxLoadedModel, TransRoleBoxLoadedModel

from utils.compute_ranks import *

logging.basicConfig(level=logging.INFO)
int_approx = None
use_bias = None
num_components = 1

num_empty_intersection = 0


def main():
    evaluate('GALEN', 'prediction', model_name='boxsqel', embedding_size=200, best=True)


def evaluate(dataset, task, model_name, embedding_size, parameters, best=True, split='test', others={}):
    global int_approx, use_bias, num_empty_intersection, num_components
    num_empty_intersection = 0
    device = get_device()

    int_approx = others['int_approx'] if 'int_approx' in others else 'none'
    use_bias = others['use_bias'] if 'use_bias' in others else False
    num_components = others['num_components'] if 'num_components' in others else 1

    model = LoadedModel.from_name(model_name, f'data/{dataset}/{task}/{model_name}', embedding_size, device, best)
    num_classes = model.class_embeds.shape[0] if model_name != 'boxel' else model.min_embedding.shape[0]

    print('Loading data')
    data_loader = DataLoader.from_task(task)
    _, classes, relations = data_loader.load_data(dataset)
    assert (len(classes) == num_classes)
    if split == 'test':
        test_data = data_loader.load_test_data(dataset, classes)
    elif split == 'val':
        test_data = data_loader.load_val_data(dataset, classes)
    else:
        raise ValueError('Unknown split.')

    # add start index for datasets with concept classes
    # this is used for the dataset where we translate r(a, b) to nf3 axiom {a} \sqsubseteq \exists r.{b}
    dataset2concept = {'natpro_abox': 9466, 'carcinogenesis_abox': 144, 'OWL2EL-2_abox': 132}
    start_index = dataset2concept[dataset] if dataset in dataset2concept else 0
    print('start_index:', start_index)

    if task == 'prediction':
        nfs = ['nf1', 'nf2', 'nf3', 'nf4']
    elif task == 'inferences':
        nfs = ['nf1']
    else:
        nfs = ['nf3']

    rankings = []
    for nf in nfs:
        ranking = compute_ranks(model, test_data, num_classes, nf, device, use_tqdm=True, start_index=start_index)
        rankings.append(ranking)

    output = '\n'.join([f'{nf.upper()}\n=========\n{rankings[i]}\n' for (i, nf) in enumerate(nfs)])
    if len(nfs) > 1:
        rankings.append(combine_rankings(rankings, num_classes))
        output += f'\nCombined\n=========\n{rankings[-1]}\n'

    print(output)
    print(
        f'emptyIntersection(nf2):{num_empty_intersection}/{len(test_data["nf2"])}={num_empty_intersection / len(test_data["nf2"])}')
    # with open(f'results/output_{model_name}_bias(v1){transbox_use_bias}_{dataset}_{task}_ours.txt', 'a') as f:
    #     f.write(f"{dataset},{task},{model_name},{embedding_size}," +
    #             output.replace("\n", "\t").replace("=", "") + '\n')

    file_name = f'result_transbox/output_{model_name}(reg,allowEmptyset)_{dataset}_{task}_new.txt'
    with open(file_name, 'a') as f:
        csv_output = f"{dataset},{task},{model_name},{parameters},\t,"
        if others:
            csv_output += ','.join([f'{k}:{v}' for k, v in others.items()]) + ',\t'
        csv_output += ',\t'.join([ranking.to_csv() for ranking in rankings])
        # add current time in dd-mm-yyyy hh:mm:ss format
        now = datetime.datetime.now()
        csv_output += f',\t{now.strftime("%d-%m-%Y %H:%M:%S")}'
        csv_output += f',\temptyIntersection(nf2):{num_empty_intersection}/{len(test_data["nf2"])}={num_empty_intersection / len(test_data["nf2"])}'
        csv_output += '\n'
        f.write(csv_output)

    return rankings


def combine_rankings(rankings, num_classes):
    combined_ranking = RankingResult(0, 0, 0, [], 0)
    for ranking in rankings:
        combined_ranking = combined_ranking.combine(ranking)
    ranks_dict = Counter(combined_ranking.ranks)
    auc = compute_rank_roc(ranks_dict, num_classes)
    combined_ranking.auc = auc
    return combined_ranking


def compute_ranks(model, eval_data, num_classes, nf, device, batch_size=10, use_tqdm=False, start_index=0):
    # start_index is used to exclude the concept classes in the ranking of link prediction tasks (i.e., r(a, b))
    if nf not in eval_data:
        raise ValueError('Tried to evaluate model on normal form not present in the evaluation data')
    eval_data = eval_data[nf]
    eval_data = eval_data.to(device)

    top1, top10, top100 = 0, 0, 0
    ranks = []
    n = len(eval_data)
    num_batches = math.ceil(n / batch_size)

    range_fun = trange if use_tqdm else range
    for i in range_fun(num_batches):
        start = i * batch_size
        current_batch_size = min(batch_size, n - start)
        batch_data = eval_data[start:start + current_batch_size, :]
        fun = f'compute_{nf}_ranks'
        if model.is_translational() and nf in ['nf3', 'nf4']:
            fun += '_translational'
            # fun += '_abox_translational'
        elif isinstance(model, TransRoleBoxLoadedModel) and nf in ['nf2', 'nf3', 'nf4']:
            fun += '_transrolebox'
        elif isinstance(model, TransBoxLoadedModel) and nf in ['nf2', 'nf3', 'nf4']:
            fun += '_transbox'
        elif isinstance(model, BoxELLoadedModel):
            fun += '_boxel'

        if 'abox' in fun:
            batch_ranks = globals()[fun](model, batch_data, current_batch_size,
                                         start_index)  # call the correct function based on NF
        else:
            batch_ranks = globals()[fun](model, batch_data, current_batch_size)

        top1 += (batch_ranks <= 1).sum()
        top10 += (batch_ranks <= 10).sum()
        top100 += (batch_ranks <= 100).sum()
        ranks += batch_ranks.tolist()

    ranks_dict = Counter(ranks)
    auc = compute_rank_roc(ranks_dict, num_classes)
    return RankingResult(top1.item(), top10.item(), top100.item(), ranks, auc)


# todo: debug this function
def compute_nf1_ranks_abox(model, batch_data, batch_size, start_index=0):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    batch_centers = centers[batch_data[:, 0]]
    # NOTE that we exclude concept classes because here we assume all nf1 axioms are translated from r(a, b)
    class_centers = centers[:start_index, ...]  # keep only the first num_concept classes

    dists = batch_centers[:, None, :] - torch.tile(class_centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    # NOTE that there is no c<=c case in the abox prediction task
    # dists.scatter_(1, batch_data[:, 0].reshape(-1, 1)-start_index, torch.inf)  # filter out c <= c
    return dists_to_ranks(dists, batch_data[:, 1])


def compute_nf1_ranks(model, batch_data, batch_size, start_index=0):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    batch_centers = centers[batch_data[:, 0]]

    dists = batch_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c <= c
    return dists_to_ranks(dists, batch_data[:, 1])


def compute_nf1_ranks_boxel(model, batch_data, batch_size, start_index=0):
    batch_mins = model.min_embedding[batch_data[:, 0]]
    batch_deltas = model.delta_embedding[batch_data[:, 0]]
    batch_maxs = batch_mins + torch.exp(batch_deltas)

    all_mins = torch.tile(model.min_embedding, (batch_size, 1, 1))  # 100x23142x200
    all_maxs = torch.tile(model.min_embedding + torch.exp(model.delta_embedding), (batch_size, 1, 1))

    inter_min = torch.max(batch_mins[:, None, :], all_mins)
    inter_max = torch.min(batch_maxs[:, None, :], all_maxs)
    inter_delta = inter_max - inter_min
    inter_volumes = F.softplus(inter_delta).prod(2)
    log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

    probs = torch.exp(log_intersection)  # 100x23142
    dists = 1 - probs
    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c <= c
    return dists_to_ranks(dists, batch_data[:, 1])


def compute_nf2_ranks_abox(model, batch_data, batch_size, start_index=0):
    class_boxes = model.get_boxes(model.class_embeds)
    c_boxes = class_boxes[batch_data[:, 0]]
    d_boxes = class_boxes[batch_data[:, 1]]

    centers = class_boxes.centers
    centers = centers[:start_index, ...]  # keep only the first num_concept classes

    intersection, _, _ = c_boxes.intersect(d_boxes)
    dists = intersection.centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)

    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c n d <= d
    dists.scatter_(1, batch_data[:, 1].reshape(-1, 1), torch.inf)  # filter out c n d <= d

    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf2_ranks(model, batch_data, batch_size, start_index=0):
    global num_empty_intersection
    class_boxes = model.get_boxes(model.class_embeds)
    c_boxes = class_boxes[batch_data[:, 0]]
    d_boxes = class_boxes[batch_data[:, 1]]

    centers = class_boxes.centers

    intersection, lower, upper = c_boxes.intersect(d_boxes)
    dists = intersection.centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)

    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c n d <= d
    dists.scatter_(1, batch_data[:, 1].reshape(-1, 1), torch.inf)  # filter out c n d <= d

    # To make sure that when the intersection is empty, the distance is all 0.
    # We set dists as 0 where lower has coordinates bigger than upper
    dists = torch.where(torch.any(lower > upper, dim=1)[:, None], torch.zeros_like(dists), dists)
    num_empty_intersection += torch.sum(torch.any(lower > upper, dim=1)).item()

    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf2_ranks_transrolebox(model, batch_data, batch_size):
    global int_approx, num_empty_intersection, num_components
    class_boxes = model.get_boxes(model.class_embeds)
    c_boxes = class_boxes[batch_data[:, 0]]
    d_boxes = class_boxes[batch_data[:, 1]]

    centers = class_boxes.centers

    if int_approx == 'none':
        intersection, _, _ = c_boxes.intersect(d_boxes)
        intersection_center = intersection.centers
    elif int_approx == 'allowEmpty':
        intersection, startAll, endAll = c_boxes.intersect(d_boxes)

        # mask = torch.where(startAll > endAll, torch.zeros_like(startAll), torch.ones_like(startAll))

        if num_components > 0:
            mask_0 = torch.where(startAll > endAll, torch.zeros_like(startAll), torch.ones_like(startAll))
            mask_1 = mask_0.reshape(mask_0.shape[0], num_components, -1)
            mask_1 = mask_1 * mask_1.min(dim=2, keepdim=True)[0]
            mask = mask_1.reshape(mask_0.shape[0], -1)
        else:
            mask = torch.where(startAll > endAll, torch.zeros_like(startAll), torch.ones_like(startAll))

        intersection_center = intersection.centers
        num_empty_intersection += torch.sum(torch.all(mask < 0.5, dim=1)).item()
    else:
        c1 = c_boxes.centers
        c2 = c_boxes.offsets

        d1 = d_boxes.centers
        d2 = d_boxes.offsets
        if int_approx == 'avg':
            intersection_center = (c1 + d1) / 2
        else:
            intersection_center = (c2 * c1 + d2 * d1) / (c2 + d2)

    dists = intersection_center[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    if int_approx == 'allowEmpty':
        dists = dists * mask[:, None, :]
    dists = torch.linalg.norm(dists, dim=2, ord=2)

    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c n d <= d
    dists.scatter_(1, batch_data[:, 1].reshape(-1, 1), torch.inf)  # filter out c n d <= d

    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf2_ranks_transbox(model, batch_data, batch_size):
    global int_approx, num_empty_intersection, num_components
    class_boxes = model.get_boxes(model.class_embeds)
    c_boxes = class_boxes[batch_data[:, 0]]
    d_boxes = class_boxes[batch_data[:, 1]]

    centers = class_boxes.centers

    intersection, startAll, endAll = c_boxes.intersect(d_boxes)

    if num_components>0:
        mask_0 = torch.where(startAll > endAll, torch.zeros_like(startAll), torch.ones_like(startAll))
        mask_1 = mask_0.reshape(mask_0.shape[0], num_components, -1)
        mask_1 = mask_1 * mask_1.min(dim=2, keepdim=True)[0]
        mask = mask_1.reshape(mask_0.shape[0], -1)

        intersection_center = intersection.centers
        num_empty_intersection += torch.sum(torch.all(mask < 0.5, dim=1)).item()

        dists = intersection_center[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
        dists = dists * mask[:, None, :]
        dists = torch.linalg.norm(dists, dim=2, ord=2)
    else:
        intersection_center = intersection.centers
        dists = intersection_center[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
        dists = torch.linalg.norm(dists, dim=2, ord=2)

    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c n d <= d
    dists.scatter_(1, batch_data[:, 1].reshape(-1, 1), torch.inf)  # filter out c n d <= d

    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf2_ranks_boxel(model, batch_data, batch_size, start_index=0):
    c_mins = model.min_embedding[batch_data[:, 0]]
    c_deltas = model.delta_embedding[batch_data[:, 0]]
    c_maxs = c_mins + torch.exp(c_deltas)

    d_mins = model.min_embedding[batch_data[:, 1]]
    d_deltas = model.delta_embedding[batch_data[:, 1]]
    d_maxs = d_mins + torch.exp(d_deltas)

    all_mins = torch.tile(model.min_embedding, (batch_size, 1, 1))  # 100x23142x200
    all_maxs = torch.tile(model.min_embedding + torch.exp(model.delta_embedding), (batch_size, 1, 1))

    inter_min1 = torch.max(c_mins, d_mins)  # compute intersection between C and D
    inter_max1 = torch.min(c_maxs, d_maxs)

    inter_min = torch.max(inter_min1[:, None, :], all_mins)  # compute intersection between (C n D) and E
    inter_max = torch.min(inter_max1[:, None, :], all_maxs)
    inter_delta = inter_max - inter_min
    inter_volumes = F.softplus(inter_delta).prod(2)
    log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

    probs = torch.exp(log_intersection)  # 100x23142
    dists = 1 - probs
    dists.scatter_(1, batch_data[:, 0].reshape(-1, 1), torch.inf)  # filter out c n d <= c
    dists.scatter_(1, batch_data[:, 1].reshape(-1, 1), torch.inf)  # filter out c n d <= d
    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf3_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    bumps = model.bumps
    head_boxes = model.get_boxes(model.relation_heads)
    tail_boxes = model.get_boxes(model.relation_tails)

    centers = class_boxes.centers
    d_centers = centers[batch_data[:, 2]]
    d_bumps = bumps[batch_data[:, 2]]
    batch_heads = head_boxes.centers[batch_data[:, 1]]
    batch_tails = tail_boxes.centers[batch_data[:, 1]]

    bumped_c_centers = torch.tile(centers, (batch_size, 1, 1)) + d_bumps[:, None, :]
    bumped_d_centers = d_centers[:, None, :] + torch.tile(bumps, (batch_size, 1, 1))

    c_dists = bumped_c_centers - batch_heads[:, None, :]
    c_dists = torch.linalg.norm(c_dists, dim=2, ord=2)
    d_dists = bumped_d_centers - batch_tails[:, None, :]
    d_dists = torch.linalg.norm(d_dists, dim=2, ord=2)
    dists = c_dists + d_dists
    return dists_to_ranks(dists, batch_data[:, 0])


def compute_nf4_ranks(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    bumps = model.bumps
    head_boxes = model.get_boxes(model.relation_heads)

    centers = class_boxes.centers
    c_bumps = bumps[batch_data[:, 1]]
    batch_heads = head_boxes.centers[batch_data[:, 0]]

    translated_heads = batch_heads - c_bumps
    dists = translated_heads[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf3_ranks_abox_translational(model, batch_data, batch_size, start_index=0):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    d_centers = centers[batch_data[:, 2]]

    # NOTE that we exclude concept classes because here we assume all nf3 axioms are translated from r(a, b)
    centers = centers[start_index:, ...]  # exclude the first num_concept classes

    batch_relations = model.relation_embeds[batch_data[:, 1]]

    translated_centers = d_centers - batch_relations
    dists = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists,
                          batch_data[:, 0] - start_index)  # subtract start_index to get the correct instance index


def compute_nf3_ranks_translational(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    d_centers = centers[batch_data[:, 2]]

    batch_relations = model.relation_embeds[batch_data[:, 1]]

    translated_centers = d_centers - batch_relations
    dists = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)

    return dists_to_ranks(dists, batch_data[:, 0])


def compute_nf3_ranks_transbox(model, batch_data, batch_size, start_index=0):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    d_centers = centers[batch_data[:, 2]]

    relation_boxes = model.get_boxes(model.relation_embeds)
    batch_relations = relation_boxes.centers[batch_data[:, 1]]

    translated_centers = d_centers + batch_relations

    if use_bias:
        bias = model.bias_embeds
        # version 1
        centers = centers + bias

        # # version 2
        # batch_biases2 = model.bias_embeds[batch_data[:, 2]]
        # centers = centers + bias
        # translated_centers = translated_centers - batch_biases2

    dists = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 0])


def compute_nf3_ranks_boxel(model, batch_data, batch_size, start_index=0):
    batch_mins = model.min_embedding[batch_data[:, 2]]
    batch_deltas = model.delta_embedding[batch_data[:, 2]]
    batch_maxs = batch_mins + torch.exp(batch_deltas)

    all_mins = torch.tile(model.min_embedding, (batch_size, 1, 1))  # 100x23142x200
    all_maxs = torch.tile(model.min_embedding + torch.exp(model.delta_embedding), (batch_size, 1, 1))
    relations = model.relation_embedding[batch_data[:, 1]]
    scalings = model.scaling_embedding[batch_data[:, 1]]
    translated_mins = all_mins * (scalings[:, None, :] + 1e-8) + relations[:, None, :]
    translated_maxs = all_maxs * (scalings[:, None, :] + 1e-8) + relations[:, None, :]

    inter_min = torch.max(batch_mins[:, None, :], translated_mins)
    inter_max = torch.min(batch_maxs[:, None, :], translated_maxs)
    inter_delta = inter_max - inter_min
    inter_volumes = F.softplus(inter_delta).prod(2)
    log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

    batch_volumes = F.softplus(translated_maxs - translated_mins).prod(2)
    log_box2 = torch.log(torch.clamp(batch_volumes, 1e-10, 1e4))

    probs = torch.exp(log_intersection - log_box2)  # 100x23142
    dists = 1 - probs
    return dists_to_ranks(dists, batch_data[:, 0])


def compute_nf4_ranks_translational(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    c_centers = centers[batch_data[:, 1]]
    batch_relations = model.relation_embeds[batch_data[:, 0]]

    translated_centers = c_centers - batch_relations
    dists = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf4_ranks_transbox(model, batch_data, batch_size, start_index=0):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    c_centers = centers[batch_data[:, 1]]

    relation_boxes = model.get_boxes(model.relation_embeds)
    batch_relations = relation_boxes.centers[batch_data[:, 0]]

    translated_centers = c_centers + batch_relations
    dists = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 2])


def compute_nf4_ranks_boxel(model, batch_data, batch_size, start_index=0):
    batch_mins = model.min_embedding[batch_data[:, 1]]
    batch_deltas = model.delta_embedding[batch_data[:, 1]]
    batch_maxs = batch_mins + torch.exp(batch_deltas)
    relations = model.relation_embedding[batch_data[:, 0]]
    scalings = model.scaling_embedding[batch_data[:, 0]]
    translated_mins = (batch_mins - relations) / (scalings + 1e-8)
    translated_maxs = (batch_maxs - relations) / (scalings + 1e-8)

    all_mins = torch.tile(model.min_embedding, (batch_size, 1, 1))  # 100x23142x200
    all_maxs = torch.tile(model.min_embedding + torch.exp(model.delta_embedding), (batch_size, 1, 1))

    inter_min = torch.max(translated_mins[:, None, :], all_mins)
    inter_max = torch.min(translated_maxs[:, None, :], all_maxs)
    inter_delta = inter_max - inter_min
    inter_volumes = F.softplus(inter_delta).prod(2)
    log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

    probs = torch.exp(log_intersection)  # 100x23142
    dists = 1 - probs
    return dists_to_ranks(dists, batch_data[:, 0])


if __name__ == '__main__':
    main()
