import torch
import numpy as np

def dists_to_ranks(dists, targets):
    index = torch.argsort(dists, dim=1).argsort(dim=1) + 1
    return torch.take_along_dim(index, targets.reshape(-1, 1), dim=1).flatten()

def compute_rank_roc(ranks, num_classes, num_negatives=None):
    sorted_ranks = sorted(list(ranks.keys()))
    tprs = [0]
    fprs = [0]
    tpr = 0
    num_triples = sum(ranks.values())
    if not num_negatives:
        num_negatives = (num_classes - 1) * num_triples
    for x in sorted_ranks:
        tpr += ranks[x]
        tprs.append(tpr / num_triples)
        fp = sum([(x - 1) * v if k <= x else x * v for k, v in ranks.items()])
        fprs.append(fp / num_negatives)

    tprs.append(1)
    fprs.append(1)
    auc = np.trapz(tprs, fprs)
    return auc



def compute_nf3_ranks_transrolebox(model, batch_data, batch_size):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    d_centers = centers[batch_data[:, 2]]

    relation_boxes = model.get_boxes(model.relation_embeds)
    batch_relations = relation_boxes.centers[batch_data[:, 1]]

    translated_centers = d_centers - batch_relations

    dists = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 0])


def compute_nf4_ranks_transrolebox(model, batch_data, batch_size, start_index=0):
    class_boxes = model.get_boxes(model.class_embeds)
    centers = class_boxes.centers
    c_centers = centers[batch_data[:, 1]]

    relation_boxes = model.get_boxes(model.relation_embeds)
    batch_relations = relation_boxes.centers[batch_data[:, 0]]

    translated_centers = c_centers - batch_relations
    dists = translated_centers[:, None, :] - torch.tile(centers, (batch_size, 1, 1))
    dists = torch.linalg.norm(dists, dim=2, ord=2)
    return dists_to_ranks(dists, batch_data[:, 2])