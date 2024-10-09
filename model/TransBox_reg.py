# Code adapted from https://github.com/bio-ontology-research-group/EL2Box_embedding.

import numpy as np
import torch.nn as nn
import torch
import os
from torch.nn.functional import relu

from model.loaded_models import TransBoxLoadedModel


class TransBox(nn.Module):
    '''
    Args:
        classNum: number of classes
        relationNum: number of relations
        embedding_dim: the dimension of the embedding(both class and relatio)
        margin: the distance that two box apart
    '''

    def __init__(self, device, class_, relationNum, embedding_dim, margin=0, num_neg=1, neg_dist=1, reg_factor=0.1,
                 vis_loss=False,
                 use_bias=False,
                 num_components=1):
        super(TransBox, self).__init__()

        self.name = 'transbox'
        self.margin = margin
        self.classNum = len(class_)
        self.class_ = class_
        self.relationNum = relationNum
        self.device = device
        self.reg_norm = 1
        self.inf = 4
        self.beta = None
        self.ranking_fn = 'l2'
        self.negative_sampling = False

        self.enhance = True

        self.num_neg = num_neg
        self.neg_dist = neg_dist
        self.use_bias = use_bias
        self.reg_factor = reg_factor
        self.num_components = num_components

        self.vis_loss = vis_loss

        self.classEmbeddingDict = nn.Embedding(self.classNum, embedding_dim * 2)
        nn.init.uniform_(self.classEmbeddingDict.weight, a=-1, b=1)
        self.classEmbeddingDict.weight.data /= torch.linalg.norm(self.classEmbeddingDict.weight.data, axis=1).reshape(
            -1, 1)

        if use_bias:
            self.biasEmbeddingDict = nn.Embedding(self.classNum, embedding_dim)
            nn.init.uniform_(self.biasEmbeddingDict.weight, a=-1, b=1)
            self.biasEmbeddingDict.weight.data /= torch.linalg.norm(self.biasEmbeddingDict.weight.data, axis=1).reshape(
                -1, 1)
        else:
            self.biasEmbeddingDict = nn.Embedding(self.classNum, 1)

        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim * 2)
        nn.init.uniform_(self.relationEmbeddingDict.weight, a=-1, b=1)
        self.relationEmbeddingDict.weight.data /= torch.linalg.norm(
            self.relationEmbeddingDict.weight.data, axis=1).reshape(-1, 1)

        self.embedding_dim = embedding_dim

    # cClass isSubSetof dClass

    def reg(self, x):
        # return 0
        res = torch.abs(torch.linalg.norm(x, axis=1) - 1)
        res = torch.reshape(res, [-1, 1])
        return res

    def nf1Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]

        c2 = torch.abs(c[:, self.embedding_dim:])
        d2 = torch.abs(d[:, self.embedding_dim:])

        # box

        cr = torch.abs(c2)
        dr = torch.abs(d2)

        cen1 = c1
        cen2 = d1
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(euc + cr - dr - self.margin), axis=1), [-1, 1])
        return dst + self.reg(cen1) + self.reg(cen2)

    # cClass and dCLass isSubSetof eClass
    def nf2Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])
        e = self.classEmbeddingDict(input[:, 2])
        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]
        e1 = e[:, :self.embedding_dim]

        c2 = torch.abs(c[:, self.embedding_dim:])
        d2 = torch.abs(d[:, self.embedding_dim:])
        e2 = torch.abs(e[:, self.embedding_dim:])

        startAll = torch.maximum(c1 - c2, d1 - d2)
        endAll = torch.minimum(c1 + c2, d1 + d2)

        if self.num_components > 0:
            dst = torch.linalg.norm(relu(startAll - endAll), axis=1)

            newR = (startAll - endAll) / 2
            cen1 = (startAll + endAll) / 2

            # introduce a new 0,1 tensor that is 0 where startAll > endAll
            mask_0 = torch.where(startAll > endAll, torch.zeros_like(startAll), torch.ones_like(startAll))
            mask_1 = mask_0.reshape(mask_0.shape[0], self.num_components, -1)
            mask_1 = mask_1 * mask_1.min(dim=2, keepdim=True)[0]
            mask = mask_1.reshape(mask_0.shape[0], -1)

            er = torch.abs(e2)

            cen2 = e1
            euc = torch.abs(cen1 - cen2)
            dist_box = (euc + newR - er) * mask
            dst = dst + torch.reshape(torch.linalg.norm(relu(dist_box - self.margin), axis=1), [-1, 1])
        else:
            newR = torch.abs(startAll - endAll) / 2

            er = torch.abs(e2)

            cen1 = (startAll + endAll) / 2
            cen2 = e1
            euc = torch.abs(cen1 - cen2)

            dst = torch.reshape(torch.linalg.norm(relu(euc + newR - er - self.margin), axis=1), [-1, 1]) \
                  + torch.linalg.norm(relu(startAll - endAll), axis=1)

        return dst + self.reg(c1) + self.reg(d1) + self.reg(e1)

    def disJointLoss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        c1 = c[:, :self.embedding_dim]
        d1 = d[:, :self.embedding_dim]

        c2 = torch.abs(c[:, self.embedding_dim:])
        d2 = torch.abs(d[:, self.embedding_dim:])

        # box

        cr = torch.abs(c2)
        dr = torch.abs(d2)

        cen1 = c1
        cen2 = d1
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(-euc + cr + dr - self.margin), axis=1), [-1, 1])

        return dst + self.reg(c1) + self.reg(d1)

    def nf3Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c1_o = c[:, :self.embedding_dim]
        c2_o = c[:, self.embedding_dim:]

        d1_o = d[:, :self.embedding_dim]
        d2_o = d[:, self.embedding_dim:]

        c1 = c1_o
        c2 = c2_o

        d1 = d1_o
        d2 = d2_o

        r1 = r[:, :self.embedding_dim]
        r2 = r[:, self.embedding_dim:]

        cr = torch.abs(c2)
        dr = torch.abs(d2)
        rr = torch.abs(r2)

        if not self.enhance:
            cen1 = r1+d1
            rdr = dr+rr
            dst = torch.reshape(torch.linalg.norm(relu(cen1 + cr - rdr - self.margin), axis=1), [-1, 1])
        else:
            cen1 = c1 - d1
            if self.use_bias:
                #  version 1
                bias_c = self.biasEmbeddingDict(input[:, 0])
                cen1 = cen1 + bias_c  # + bias_d

                # # version 2
                # bias_c = self.biasEmbeddingDict(input[:, 0])
                # bias_d = self.biasEmbeddingDict(input[:, 2])
                # cen1 = cen1 + bias_c + bias_d

            cdr = cr + dr
            cen2 = r1
            euc = torch.abs(cen1 - cen2)

            dst = torch.reshape(torch.linalg.norm(relu(euc + cdr - rr - self.margin), axis=1), [-1, 1])

        return dst + self.reg(c1_o) + self.reg(d1_o)

    def neg_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        c1_o = c[:, :self.embedding_dim]
        c2_o = c[:, self.embedding_dim:]

        d1_o = d[:, :self.embedding_dim]
        d2_o = d[:, self.embedding_dim:]

        c1 = c1_o
        c2 = c2_o

        d1 = d1_o
        d2 = d2_o

        cr = torch.abs(c2)
        dr = torch.abs(d2)

        r1 = r[:, :self.embedding_dim]
        r2 = r[:, self.embedding_dim:]
        rr = torch.abs(r2)

        if not self.enhance:
            cen1 = r1+d1
            rdr = dr+rr
            dst = torch.reshape(torch.linalg.norm(relu(-cen1 + cr + rdr - self.margin), axis=1), [-1, 1])
        else:
            cen1 = c1 - d1
            if self.use_bias:
                # version 1
                bias_c = self.biasEmbeddingDict(input[:, 0])
                cen1 = cen1 + bias_c  # + bias_d

                # # version 2
                # bias_c = self.biasEmbeddingDict(input[:, 0])
                # bias_d = self.biasEmbeddingDict(input[:, 2])
                # cen1 = cen1 + bias_c + bias_d

            cdr = cr + dr
            cen2 = r1
            euc = torch.abs(cen1 - cen2)

            dst = torch.reshape(torch.linalg.norm(relu(-(euc - cdr - rr) + self.margin), axis=1), [-1, 1])


        return dst + self.reg(c1_o) + self.reg(d1_o)

    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        c = self.classEmbeddingDict(input[:, 1])

        r = self.relationEmbeddingDict(input[:, 0])

        d = self.classEmbeddingDict(input[:, 2])

        c1_o = c[:, :self.embedding_dim]
        c2_o = c[:, self.embedding_dim:]

        d1_o = d[:, :self.embedding_dim]
        d2_o = d[:, self.embedding_dim:]

        c1 = c1_o
        c2 = c2_o

        d1 = d1_o
        d2 = d2_o

        cr = torch.abs(c2)
        dr = torch.abs(d2)

        r1 = r[:, :self.embedding_dim]
        r2 = r[:, self.embedding_dim:]
        rr = torch.abs(r2)

        cen1 = c1 + r1
        # if self.use_bias:
        #     bias_c = self.biasEmbeddingDict(input[:, 1])
        #     #bias_d = self.biasEmbeddingDict(input[:, 2])
        #     cen1 = cen1 - bias_c # - bias_d
        # crr = relu(rr - cr)
        crr = rr + cr
        cen2 = d1
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(euc + crr - dr - self.margin), axis=1),
                            [-1, 1])

        return dst + self.reg(c1_o) + self.reg(d1_o)

    def role_inclusion_loss(self, input):
        r = self.relationEmbeddingDict(input[:, 0])
        s = self.relationEmbeddingDict(input[:, 1])

        r_o = r[:, :self.embedding_dim]
        r_t = torch.abs(r[:, self.embedding_dim:])

        s_o = s[:, :self.embedding_dim]
        s_t = torch.abs(s[:, self.embedding_dim:])

        cen1 = r_o
        cen2 = s_o
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(euc + r_t - s_t - self.margin), axis=1), [-1, 1])

        return dst

    def role_chain_loss(self, input):
        r = self.relationEmbeddingDict(input[:, 0])
        s = self.relationEmbeddingDict(input[:, 1])
        t = self.relationEmbeddingDict(input[:, 2])

        r_o = r[:, :self.embedding_dim]
        r_t = torch.abs(r[:, self.embedding_dim:])

        s_o = s[:, :self.embedding_dim]
        s_t = torch.abs(s[:, self.embedding_dim:])

        t_o = t[:, :self.embedding_dim]
        t_t = torch.abs(t[:, self.embedding_dim:])

        cen1 = r_o + s_o
        cen2 = t_o
        euc = torch.abs(cen1 - cen2)

        dst = torch.reshape(torch.linalg.norm(relu(euc + r_t + s_t - t_t - self.margin), axis=1), [-1, 1])
        return dst

    def forward(self, input):
        batch = 1024

        # nf1
        if len(input['nf1']) == 0:
            loss1 = 0
        else:
            rand_index = np.random.choice(len(input['nf1']), size=batch)
            nf1Data = input['nf1'][rand_index]
            nf1Data = nf1Data.to(self.device)
            loss1 = self.nf1Loss(nf1Data).square().mean()

        # nf2
        if len(input['nf2']) == 0:
            loss2 = 0
        else:
            rand_index = np.random.choice(len(input['nf2']), size=batch)
            nf2Data = input['nf2'][rand_index]
            nf2Data = nf2Data.to(self.device)
            loss2 = self.nf2Loss(nf2Data).square().mean()

        # nf3
        if len(input['nf3']) == 0:
            loss3 = 0
        else:
            rand_index = np.random.choice(len(input['nf3']), size=batch)
            nf3Data = input['nf3'][rand_index]
            nf3Data = nf3Data.to(self.device)
            loss3 = self.nf3Loss(nf3Data).square().mean()

        # nf4
        if len(input['nf4']) == 0:
            loss4 = 0
        else:
            rand_index = np.random.choice(len(input['nf4']), size=batch)
            nf4Data = input['nf4'][rand_index]
            nf4Data = nf4Data.to(self.device)
            loss4 = self.nf4Loss(nf4Data).square().mean()

        # disJoint
        if len(input['disjoint']) == 0:
            disJointLoss = 0
        else:
            rand_index = np.random.choice(len(input['disjoint']), size=batch)
            disJointData = input['disjoint'][rand_index]
            disJointData = disJointData.to(self.device)
            disJointLoss = self.disJointLoss(disJointData).square().mean()

        # negLoss
        negLoss = 0
        for i in range(self.num_neg):
            if len(input[f'nf3_neg{i}']) > 0:
                rand_index = np.random.choice(len(input[f'nf3_neg{i}']), size=batch)
                negData = input[f'nf3_neg{i}'][rand_index]
                negData = negData.to(self.device)
                negLoss = negLoss + (self.neg_dist - self.neg_loss(negData)).square().mean()
            negLoss = negLoss / self.num_neg

        if 'role_inclusion' in input and len(input['role_inclusion']) > 0:
            rand_index = np.random.choice(len(input['role_inclusion']), size=batch)
            riData = input['role_inclusion'][rand_index]
            riData = riData.to(self.device)
            ri_loss = self.role_inclusion_loss(riData).square().mean()
        else:
            ri_loss = 0

        if 'role_chain' in input and len(input['role_chain']) > 0:
            rand_index = np.random.choice(len(input['role_chain']), size=batch)
            rcData = input['role_chain'][rand_index]
            rcData = rcData.to(self.device)
            rc_loss = self.role_chain_loss(rcData).square().mean()
        else:
            rc_loss = 0

        if self.vis_loss:
            vis_loss = torch.relu(.2 - torch.abs(self.classEmbeddingDict.weight[:, self.embedding_dim:]))
            vis_loss = vis_loss.mean()
        else:
            vis_loss = 0

        reg_loss = 0

        # require the offset length of relation smaller than 1
        relationOffsetNorm = torch.linalg.norm(self.relationEmbeddingDict.weight[:, self.embedding_dim:], dim=1)
        reg_loss = reg_loss + (self.reg_factor * relu(relationOffsetNorm - 1).mean())
        #
        # classCenterNorm = torch.linalg.norm(self.classEmbeddingDict.weight[:, :self.embedding_dim], dim=1)
        # reg_loss = reg_loss + (self.reg_factor * (classCenterNorm - 1).square().mean())

        if self.use_bias:
            reg_bias = torch.linalg.norm(self.biasEmbeddingDict.weight, dim=1).mean()
            reg_loss = reg_loss + self.reg_factor * reg_bias

        totalLoss = [
            loss1 + loss2 + disJointLoss + loss3 + loss4 + negLoss + vis_loss + ri_loss + rc_loss + reg_loss
        ]

        return sum(totalLoss)

    def to_loaded_model(self):
        model = TransBoxLoadedModel()
        model.embedding_size = self.embedding_dim
        model.class_embeds = self.classEmbeddingDict.weight.detach()
        model.relation_embeds = self.relationEmbeddingDict.weight.detach()
        model.bias_embeds = self.biasEmbeddingDict.weight.detach()
        return model

    def save(self, folder, best=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        suffix = '_best' if best else ''
        np.save(f'{folder}/class_embeds{suffix}.npy', self.classEmbeddingDict.weight.detach().cpu().numpy())
        np.save(f'{folder}/relations{suffix}.npy', self.relationEmbeddingDict.weight.detach().cpu().numpy())
        np.save(f'{folder}/bias{suffix}.npy', self.biasEmbeddingDict.weight.detach().cpu().numpy())
