from collections import Counter

import logging
import numpy as np
import torch
import torch.nn.functional as F
import math
from tqdm import trange
from boxes import Boxes
import json

from ranking_result import RankingResult
from utils.utils import get_device
from utils.data_loader import DataLoader
from model.loaded_models import LoadedModel, BoxELLoadedModel, TransBoxLoadedModel, TransRoleBoxLoadedModel, \
    ElbeLoadedModel

from utils.compute_ranks import *

logging.basicConfig(level=logging.INFO)
all_complex_box_centers = []
all_complex_box_offsets = []
all_complex_box_masks = []
int_approx = 'none'
num_A2C = 0
num_C2A = 0
C2D, A2C, C2A = {}, {}, {}


def main(dataset, model_name, embedding_size):
    evaluate_axiom_learning(dataset, 'prediction', model_name=model_name, embedding_size=embedding_size, best=True)


def evaluate_axiom_learning(dataset, task, model_name, embedding_size, parameters={}, best=True, others={}):
    global int_approx, num_A2C, num_C2A, C2D, A2C, C2A, \
        all_complex_box_centers, all_complex_box_masks, all_complex_box_offsets
    device = get_device()
    num_A2C, num_C2A = 0, 0
    all_complex_box_centers = []
    all_complex_box_offsets = []
    all_complex_box_masks = []
    C2D, A2C, C2A = {}, {}, {}

    if model_name == 'elbe' or model_name == 'boxel':
        int_approx = 'none'
    else:
        int_approx = 'allowEmpty'

    # int_approx = others['int_approx'] if 'int_approx' in others else 'allowEmpty'

    print(f'data/{dataset}/{task}/{model_name}')
    model = LoadedModel.from_name(model_name, f'data/{dataset}/{task}/{model_name}', embedding_size, device, best)
    num_classes = model.class_embeds.shape[0] if model_name != 'boxel' else model.min_embedding.shape[0]

    print('Loading data')
    data_path = f'data/{dataset}/{dataset}_filtered_axioms_dict.txt'
    # load test data for the given path
    # each line in the file is a dictionary
    test_data = []
    with open(data_path, 'r') as f:
        for line in f:
            test_data.append(eval(line))

    Axioms_type = ['A2C', 'C2A',]

    rankings = {}
    num_empty = 0
    for at in Axioms_type:
        ranking, new_empty = compute_ranks(model, test_data, num_classes, device, use_tqdm=True, axiom_type=at)
        num_empty += new_empty
        if ranking:
            rankings[at] = ranking

    num_negatives = (num_classes - 1) * (len(C2A.keys())+len(A2C.keys()))
    num_negatives += (len(all_complex_box_centers) - 1) * (len(C2D.keys())*2+len(C2A.keys())+len(A2C.keys()))
    print(f'num_all_complex_concepts: {len(all_complex_box_centers)}, num_C2D: {len(C2D.keys())},')

    print(
        f'num_A2C: {num_A2C}, num_C2A: {num_C2A}, empty_boxes: {num_empty}/{num_C2A + num_A2C} = {num_empty / (num_C2A + num_A2C)} ')

    all_complex_box_centers = torch.stack(all_complex_box_centers, dim=0)
    all_complex_box_offsets = torch.stack(all_complex_box_offsets, dim=0)
    all_complex_box_masks = torch.stack(all_complex_box_masks, dim=0)


    print("non-empty dimension mean: ", all_complex_box_masks.sum(dim=1).mean())
    for eval_type in ['?C2*', '*2?C']:
        ranking, new_empty = compute_ranks_C2D(model, eval_type)
        print(f'empty_boxes in C2D(ranking {eval_type}): {new_empty}')
        rankings[f'C2D:{eval_type}'] = ranking


    output = '\n'.join([f'{key}\n=========\n{rankings[key]}\n' for key in rankings])
    if len(rankings) > 1:
        rankings_values = list(rankings.values())
        rankings['combined'] = combine_rankings(rankings_values, num_classes, num_negatives)
        output += f'\nCombined\n=========\n{rankings["combined"]}\n'
    print(output)

    file_name = f'output_axiom_learning.txt'
    with open(file_name, 'a') as f:
        csv_output = f"============{dataset},{task},{model_name},{parameters}===========,\t"
        if others:
            csv_output += ','.join([f'{k}:{v}' for k, v in others.items()]) + ',\t'
        csv_output += ',\t'.join([f"{key}: {value.to_csv()}" for key, value in rankings.items()])
        csv_output += '\n\n'
        f.write(csv_output.replace(",\t", "\\\\\n").replace(',', ' & '))
    if not rankings:
        return []
    else:
        return [rankings[key] for key in rankings.keys()]


def combine_rankings(rankings, num_classes, num_negatives=None):
    combined_ranking = RankingResult(0, 0, 0, [], 0)
    for ranking in rankings:
        combined_ranking = combined_ranking.combine(ranking)
    ranks_dict = Counter(combined_ranking.ranks)
    auc = compute_rank_roc(ranks_dict, num_classes, num_negatives)
    combined_ranking.auc = auc
    return combined_ranking


def compute_ranks(model, eval_data, num_classes, device, batch_size=100, use_tqdm=False, axiom_type='A2C'):
    global num_A2C, num_C2A
    # start_index is used to exclude the concept classes in the ranking of link prediction tasks (i.e., r(a, b))
    top1, top10, top100 = 0, 0, 0
    ranks = []
    n = len(eval_data)
    num_batches = math.ceil(n / batch_size)

    empty_count = 0

    range_fun = trange if use_tqdm else range
    for i in range_fun(num_batches):
        start = i * batch_size
        current_batch_size = min(batch_size, n - start)
        batch_data = eval_data[start:start + current_batch_size]
        batch_ranks, new_empty_count = compute_complex_axiom_ranks(model, batch_data, current_batch_size, device,
                                                                   axiom_type)

        empty_count += new_empty_count

        if isinstance(batch_ranks, torch.Tensor):
            top1 += (batch_ranks <= 1).sum()
            top10 += (batch_ranks <= 10).sum()
            top100 += (batch_ranks <= 100).sum()
            ranks += batch_ranks.tolist()

    if axiom_type == 'A2C':
        print(f"empty_boxes in {axiom_type}: {empty_count}/{num_A2C} = {empty_count / num_A2C}")
    else:
        print(f"empty_boxes in {axiom_type}: {empty_count}/{num_C2A} = {empty_count / num_C2A}")

    if not ranks:
        return False, empty_count

    ranks_dict = Counter(ranks)
    auc = compute_rank_roc(ranks_dict, num_classes)
    return RankingResult(top1.item(), top10.item(), top100.item(), ranks, auc), empty_count


# Function to compute the embedding for ∃r.C using an external method
def existential_embedding(role_box, concept_box, model, scaling=None):
    if isinstance(model, TransRoleBoxLoadedModel):
        new_center = concept_box.centers - role_box.centers
        new_offset = role_box.offsets + concept_box.offsets
        return Boxes(new_center, new_offset)
    elif isinstance(model, TransBoxLoadedModel):
        new_center = concept_box.centers + role_box.centers
        new_offset = role_box.offsets + concept_box.offsets
        return Boxes(new_center, new_offset)
        # new_center = role_box.centers + concept_box.centers
        # new_offset = F.relu(role_box.offsets - concept_box.offsets)
        # return Boxes(new_center, new_offset)
    elif isinstance(model, ElbeLoadedModel):
        new_center = concept_box.centers - role_box
        new_offset = concept_box.offsets
        return Boxes(new_center, new_offset)
    elif isinstance(model, BoxELLoadedModel):
        new_center = (concept_box.centers - role_box) / (scaling + 1e-8)
        new_offset = concept_box.centers / (scaling + 1e-8)
        return Boxes(new_center, new_offset)
    else:
        assert False, 'Model not supported'


# Define a custom error class
class EmptyIntersectError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def intersection_embedding(box1, box2):
    global int_approx
    if int_approx == 'none':
        intersect_boxes, lower, upper = box1.intersect(box2)
        # arise intersection_vaid_error if the lower bound is greater than the upper bound
        if not torch.all(lower <= upper):
            raise EmptyIntersectError('Intersection invalid')
        else:
            return intersect_boxes, torch.ones_like(lower)
    elif int_approx == 'allowEmpty':
        intersect_boxes, startAll, endAll = box1.intersect(box2)
        mask = torch.where(startAll > endAll, torch.zeros_like(startAll), torch.ones_like(startAll))
        return intersect_boxes, mask
    else:
        assert False, 'Model not supported'


# Function to compute the new embedding for a complex concept
def compute_new_embedding(complex_concept, concept_embeddings, role_embeddings, model, scaling_embedding=None):
    if isinstance(complex_concept, int):
        # Atomic concept: Look up the embedding from concept_embeddings
        concept_ind = complex_concept
        concept_embed = concept_embeddings[concept_ind]
        embed_size = concept_embed.size(0) // 2
        return Boxes(concept_embed[:embed_size], torch.abs(concept_embed[embed_size:])), torch.ones_like(
            concept_embed[:embed_size])

    elif "conjunct" in complex_concept:
        # Conjunction: Compute the intersection of multiple concepts
        boxes = [compute_new_embedding(sub_concept, concept_embeddings, role_embeddings, model, scaling_embedding)
                 for sub_concept in complex_concept["conjunct"]]
        # Start with the first box and intersect with each subsequent box
        result_box, mask = boxes[0]
        for new_box, new_mask in boxes[1:]:
            # result_box = result_box.intersect(box)[0]
            result_box, result_mask = intersection_embedding(result_box, new_box)
            mask = mask * result_mask * new_mask
        return result_box, mask

    elif "exists" in complex_concept:
        # Existential restriction: ∃r.C
        role_ind = complex_concept["exists"][0]
        sub_concept = complex_concept["exists"][1]

        # Look up the role embedding and the concept embedding recursively
        scaling = None
        if isinstance(model, TransBoxLoadedModel) or isinstance(model, TransRoleBoxLoadedModel):
            role_box = role_embeddings[role_ind]
            role_box = Boxes(role_box[:role_box.size(0) // 2], torch.abs(role_box[role_box.size(0) // 2:]))
        elif isinstance(model, ElbeLoadedModel):
            role_box = role_embeddings[role_ind]
        elif isinstance(model, BoxELLoadedModel):
            role_box = role_embeddings[role_ind]
            scaling = scaling_embedding[role_ind]
        else:
            assert False, 'Model not supported'

        sub_concept_box, mask = compute_new_embedding(sub_concept, concept_embeddings, role_embeddings, model,
                                                      scaling_embedding)

        # Call the external function to compute the new embedding for ∃r.C
        if isinstance(model, BoxELLoadedModel):
            assert scaling is not None
            return existential_embedding(role_box, sub_concept_box, model, scaling), mask
        else:
            return existential_embedding(role_box, sub_concept_box, model), mask

    else:
        assert False, f'Unknown complex concept: {complex_concept}'


def loading_data(model, device):
    if isinstance(model, BoxELLoadedModel):
        all_mins = model.min_embedding
        all_deltas = model.delta_embedding
        all_maxs = all_mins + torch.exp(all_deltas)

        centers = (all_mins + all_maxs) / 2
        offsets = (all_maxs - all_mins) / 2

        class_embeds = torch.cat([centers, offsets], dim=1).to(device)
        relation_embeds = model.relation_embedding.to(device)
        centers = centers.to(device)
        scalings_embeddings = model.scaling_embedding.to(device)
    else:
        class_embeds = model.class_embeds.to(device)
        relation_embeds = model.relation_embeds.to(device)

        class_boxes = model.get_boxes(class_embeds)
        centers = class_boxes.centers
        offsets = class_boxes.offsets
        scalings_embeddings = None

    return centers, offsets, class_embeds, relation_embeds, scalings_embeddings


def compute_dist(model, batch_size, centers, offsets, C_centers, C_offsets, C_masks, axiom_type):
    if isinstance(model, BoxELLoadedModel):
        all_mins = torch.tile(model.min_embedding, (batch_size, 1, 1))  # 100x23142x200
        all_maxs = torch.tile(model.min_embedding + torch.exp(model.delta_embedding), (batch_size, 1, 1))

        batch_mins = C_centers - C_offsets
        batch_maxs = C_centers + C_offsets

        inter_min = torch.max(batch_mins[:, None, :], all_mins)
        inter_max = torch.min(batch_maxs[:, None, :], all_maxs)
        inter_delta = inter_max - inter_min
        inter_volumes = F.softplus(inter_delta).prod(2)
        log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

        probs = torch.exp(log_intersection)
        dists = -probs
    else:
        dists = C_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))

        if int_approx == 'none':
            dists = torch.linalg.norm(dists, dim=2, ord=2)
        else:
            dists = torch.linalg.norm(dists * (C_masks[:, None, :]), dim=2, ord=2)

            if axiom_type == 'A2C':
                # A2C: add the sum of offset part where C is empty
                # dists += torch.linalg.norm(
                #     torch.tile(offsets, (batch_size, 1, 1)) * (1 - C_masks[:, None, :]),
                #     dim=2, ord=2)
                pass
                # dists += torch.where((1 - C_masks[:, None, :]).max(dim=2)[0] > 0.5, 1e10, dists)
                # dists += torch.linalg.norm(
                #     torch.tile(offsets, (batch_size, 1, 1)) * (1 - C_masks[:, None, :]),
                #     dim=2, ord=2)
    return dists


def record_complex_boxes(Box_CC, mask_CC, pair_A_ind_type = None):
    global all_complex_box_centers, all_complex_box_offsets, all_complex_box_masks, C2A, A2C
    all_complex_box_centers.append(Box_CC.centers)
    all_complex_box_offsets.append(Box_CC.offsets)
    all_complex_box_masks.append(mask_CC)
    assert len(all_complex_box_centers) == len(all_complex_box_offsets) == len(all_complex_box_masks)
    if pair_A_ind_type is not None:
        A_ind, data_type = pair_A_ind_type
        if data_type == 'C2A':
            C2A[len(all_complex_box_centers) - 1] = A_ind
        else:
            A2C[A_ind] = len(all_complex_box_centers) - 1

    return len(all_complex_box_masks) - 1


def record_C2D_boxes(dict, class_embeds, relation_embeds, model, scalings_embeddings):
    global C2D, all_complex_box_centers
    C_dict, D_dict = dict['subclass'], dict['superclass']
    new_inds = []
    for con_dict in [C_dict, D_dict]:
        try:
            Box_CC, mask_CC = compute_new_embedding(con_dict, class_embeds, relation_embeds, model,
                                                    scalings_embeddings)
        except EmptyIntersectError:
            zeros_tensor = torch.zeros((class_embeds.shape[1] // 2,), device=class_embeds.device)
            new_inds.append(record_complex_boxes(Boxes(zeros_tensor, zeros_tensor), zeros_tensor))
            continue

        new_inds.append(record_complex_boxes(Box_CC, mask_CC))

    C2D[new_inds[0]] = new_inds[1]


def compute_complex_axiom_ranks(model, batch_data, batch_size, device, axiom_type):
    global num_A2C, num_C2A
    centers, offsets, class_embeds, relation_embeds, scalings_embeddings = loading_data(model, device,)

    A_indices = []
    C_centers, C_offsets, C_masks = [], [], []
    empty_count = 0
    for ax_dict in batch_data:
        if axiom_type == 'A2C':
            key_A = 'subclass'
            key_C = 'superclass'
        else:
            key_A = 'superclass'
            key_C = 'subclass'

        if isinstance(ax_dict[key_A], dict):
            # ignore complex axioms C2D
            if axiom_type == 'C2A' and isinstance(ax_dict[key_C], dict):
                # only record C2D boxes when testing A2C axioms
                record_C2D_boxes(ax_dict, class_embeds, relation_embeds, model, scalings_embeddings)
            batch_size -= 1
            continue
        else:
            if axiom_type == 'A2C':
                num_A2C += 1
            else:
                num_C2A += 1

        C_dict = ax_dict[key_C]
        assert isinstance(ax_dict[key_A], int), f'Error: {ax_dict[key_A]} is not an integer'
        A_indices.append(ax_dict[key_A])

        # except the intersection_valid_error
        try:
            Box_C, mask_C = compute_new_embedding(C_dict, class_embeds, relation_embeds, model, scalings_embeddings)
        except EmptyIntersectError:
            # print("Empty intersection error on:" + str(dict))
            batch_size -= 1
            empty_count += 1
            dim = class_embeds.shape[1] // 2
            zeros_tensor = torch.zeros((dim,), device=device)
            record_complex_boxes(Boxes(zeros_tensor, zeros_tensor), zeros_tensor, tuple([ax_dict[key_A], axiom_type]))
            # A_indices[-1] = class_embeds.shape[0]//2
            continue

        record_complex_boxes(Box_C, mask_C, tuple([ax_dict[key_A], axiom_type]))

        C_centers.append(Box_C.centers)
        C_offsets.append(Box_C.offsets)
        C_masks.append(mask_C)

    A_indices = torch.tensor(A_indices, dtype=torch.long, device=device)

    if batch_size == 0:
        dists = torch.zeros((empty_count, class_embeds.shape[0]), device=device)
    else:

        C_centers = torch.stack(C_centers, dim=0)
        C_offsets = torch.stack(C_offsets, dim=0)
        C_masks = torch.stack(C_masks, dim=0)

        dists = compute_dist(model, batch_size, centers, offsets, C_centers, C_offsets, C_masks, axiom_type)
        # add zero to the distance of the empty intersection
        dists = torch.cat([dists, torch.zeros((empty_count, dists.shape[1]), device=dists.device)], dim=0)

    return dists_to_ranks(dists, A_indices), empty_count

def dist_BoxEL(batch_mins, batch_maxs, all_mins, all_maxs):
    inter_min = torch.max(batch_mins[None, :], all_mins)
    inter_max = torch.min(batch_maxs[None, :], all_maxs)
    inter_delta = inter_max - inter_min
    inter_volumes = F.softplus(inter_delta).prod(1)
    log_intersection = torch.log(torch.clamp(inter_volumes, 1e-10, 1e4))

    probs = torch.exp(log_intersection)
    dist = -probs
    return dist

def dist_C2D_otherModel(query_center, masks, eval_type):
    global all_complex_box_centers, all_complex_box_masks, all_complex_box_offsets, int_approx
    if int_approx == 'none':
        if eval_type in {"C", "C2A"}:
            dist = all_complex_box_centers - query_center
            if torch.any(masks == 0):
                return torch.zeros(all_complex_box_centers.shape[0], device=all_complex_box_centers[0].device)
            else:
                dist = torch.linalg.norm(dist, dim=1, ord=2)
                dist = torch.where(all_complex_box_masks.min(dim=1)[0] < 0.5, 1e10, dist)
        else:
            dist = query_center - all_complex_box_centers
            if torch.any(masks == 0):
                return torch.zeros(all_complex_box_centers.shape[0], device=all_complex_box_centers[0].device)
            else:
                dist = torch.linalg.norm(dist, dim=1, ord=2)
                dist = torch.where(all_complex_box_masks.min(dim=1)[0] < 0.5, 1e10, dist)

    else:
        if eval_type in {"C", "C2A"}:
            dist = all_complex_box_centers - query_center

            dist = torch.linalg.norm(dist * all_complex_box_masks * masks[None, :], dim=1, ord=2)
            dist = torch.where(
                (all_complex_box_masks * (1 - masks[None, :])).sum(dim=1) > 0.5, 1e10, dist)

            # assert dist.square().sum() != 0

        else:
            dist = query_center - all_complex_box_centers
            dist = torch.linalg.norm(dist * masks[None, :] * all_complex_box_masks, dim=1, ord=2)
            dist = torch.where(
                (masks[None, :] * (1 - all_complex_box_masks)).sum(dim=1) > 0.5, 1e10, dist)
             #assert dist.square().sum() != 0

    return dist




def dists_C2D(model, left_ind, right_ind, eval_type):
    global all_complex_box_centers, all_complex_box_masks, all_complex_box_offsets, int_approx

    if isinstance(model, BoxELLoadedModel):
        all_mins = all_complex_box_centers - torch.exp(all_complex_box_offsets)
        all_maxs = all_complex_box_centers + torch.exp(all_complex_box_offsets)

        if eval_type== 'C':
            query_min = all_mins[right_ind]
            query_max= all_maxs[right_ind]
            masks = all_complex_box_masks[right_ind]
        elif eval_type == 'D':
            query_min = all_mins[left_ind]
            query_max = all_maxs[left_ind]
            masks = all_complex_box_masks[left_ind]
        else:
            if eval_type == '?C2A':
                query_min = model.min_embedding[right_ind]
                query_delta = model.delta_embedding[right_ind]
            else:
                query_min = model.min_embedding[left_ind]
                query_delta = model.delta_embedding[left_ind]

            query_max = query_min + torch.exp(query_delta)
            masks = torch.ones_like(query_max)

        if torch.any(masks==0):
            return torch.zeros(all_complex_box_centers.shape[0], device=all_complex_box_centers[0].device)

        return dist_BoxEL(query_min, query_max, all_mins, all_maxs)
    else:
        if eval_type == 'C':
            query_centers = all_complex_box_centers[right_ind]
            query_masks = all_complex_box_masks[right_ind]
        elif eval_type == 'D':
            query_centers = all_complex_box_centers[left_ind]
            query_masks = all_complex_box_masks[left_ind]
        else:
            if eval_type == '?C2A':
                query_centers = model.get_boxes(model.class_embeds)[right_ind].centers
            else:
                query_centers = model.get_boxes(model.class_embeds)[left_ind].centers
            query_masks = torch.ones_like(query_centers)

        return dist_C2D_otherModel(query_centers, query_masks, eval_type)


def compute_dist_one_kind(model, eval_type):
    global C2D, C2A, A2C
    dists = []
    Answer_ind = []
    num_empty = 0
    if eval_type in {"C", "D"}:
        all_items = C2D.items()
    elif eval_type == "?C2A":
        all_items = C2A.items()
    else:
        all_items = A2C.items()

    for left_ind, right_ind in all_items:
        # record answer index
        if eval_type in {"C", "?C2A"}:
            Answer_ind.append(left_ind)
        else:
            Answer_ind.append(right_ind)

        dist = dists_C2D(model, left_ind, right_ind, eval_type)

        if dist.square().sum() == 0:
            # Answer_ind[-1] = len(all_complex_box_centers) - 1
            num_empty += 1

        # filter D<D or C<C
        if eval_type == 'C':
            dist[right_ind] = 1e10
        elif eval_type == 'D':
            dist[left_ind] = 1e10

        dists.append(dist)
    return dists, Answer_ind, num_empty


def compute_ranks_C2D(model, group_type ="?C2*"):
    global C2D, A2C, C2A, all_complex_box_centers, all_complex_box_offsets, all_complex_box_masks

    dists = []
    Answer_ind = []
    num_empty = 0
    if group_type == '?C2*':
        eval_types = ['?C2A', 'C']
    else:
        eval_types = ['A2?C', 'D']

    for eval_type in eval_types:
        new_dists, new_Answer_ind, new_num_empty = compute_dist_one_kind(model, eval_type)
        dists += new_dists
        Answer_ind += new_Answer_ind
        num_empty += new_num_empty

    dists = torch.stack(dists, dim=0)
    Answer_ind = torch.tensor(Answer_ind, dtype=torch.long, device=dists.device)
    print(dists.shape, Answer_ind.shape)

    ranks = dists_to_ranks(dists, Answer_ind)
    top1 = (ranks <= 1).sum()
    top10 = (ranks <= 10).sum()
    top100 = (ranks <= 100).sum()

    ranks = ranks.tolist()

    ranks_dict = Counter(ranks)
    auc = compute_rank_roc(ranks_dict, len(all_complex_box_centers))
    print(len(all_complex_box_centers))
    return RankingResult(top1.item(), top10.item(), top100.item(), ranks, auc), num_empty


if __name__ == '__main__':
    with open(f'configs_our.json', 'r') as f:
        configs = json.load(f)

    for dataset in ['GALEN', 'GO', 'ANATOMY']:
        for model_name in [ 'boxel', 'elbe', 'transrolebox', 'transbox' ]:
            print(f"==========={model_name}, {dataset}============")
            main(dataset, model_name, configs[model_name][dataset]['prediction']["embedding_dim"])
