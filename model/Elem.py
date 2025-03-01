# Code adapted from https://github.com/bio-ontology-research-group/EL2Box_embedding.

import numpy as np
import torch.nn as nn
import torch
import os

from model.loaded_models import ElbeLoadedModel


class Elem(nn.Module):

    def __init__(self, device, class_, relationNum, embedding_dim, margin=0, reg_norm=1):
        super(Elem, self).__init__()
        self.margin = margin
        self.classNum = len(class_)
        self.relationNum = relationNum
        self.device = device
        self.reg_norm = reg_norm
        self.inf = 100.0
        self.negative_sampling = False
        self.name = 'elem'

        self.classEmbeddingDict = nn.Embedding(self.classNum, embedding_dim + 1)
        nn.init.uniform_(self.classEmbeddingDict.weight, a=-1, b=1)
        self.classEmbeddingDict.weight.data /= torch.linalg.norm(self.classEmbeddingDict.weight.data, axis=1).reshape(
            -1, 1)

        self.relationEmbeddingDict = nn.Embedding(relationNum, embedding_dim)
        nn.init.uniform_(self.relationEmbeddingDict.weight, a=-1, b=1)
        self.relationEmbeddingDict.weight.data /= torch.linalg.norm(
            self.relationEmbeddingDict.weight.data, axis=1).reshape(-1, 1)

        self.embedding_dim = embedding_dim

    # cClass isSubSetof dClass
    def reg(self, x):
        res = torch.abs(torch.linalg.norm(x, axis=1) - self.reg_norm)
        res = torch.reshape(res, [-1, 1])
        return res

    def nf1Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        rc = torch.abs(c[:, -1])
        rd = torch.abs(d[:, -1])
        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]

        euc = torch.linalg.norm(x1 - x2, axis=1)
        relu = torch.nn.ReLU()
        dst = torch.reshape(relu(euc + rc - rd - self.margin), [-1, 1])

        return dst + self.reg(x1) + self.reg(x2)

    # cClass and dCLass isSubSetof eClass
    def nf2Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])
        e = self.classEmbeddingDict(input[:, 2])

        rc = torch.reshape(torch.abs(c[:, -1]), [-1, 1])
        rd = torch.reshape(torch.abs(d[:, -1]), [-1, 1])
        re = torch.reshape(torch.abs(e[:, -1]), [-1, 1])

        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        x3 = e[:, 0:-1]

        sr = rc + rd

        x = x2 - x1
        dst = torch.reshape(torch.linalg.norm(x, axis=1), [-1, 1])
        dst2 = torch.reshape(torch.linalg.norm(x3 - x1, axis=1), [-1, 1])
        dst3 = torch.reshape(torch.linalg.norm(x3 - x2, axis=1), [-1, 1])
        relu = torch.nn.ReLU()
        # rdst = relu(torch.minimum(rc, rd) - re)
        relu1 = torch.nn.ReLU()
        relu2 = torch.nn.ReLU()
        relu3 = torch.nn.ReLU()

        dst_loss = (relu1(dst - sr - self.margin)
                    + relu2(dst2 - rc - self.margin)
                    + relu3(dst3 - rd - self.margin))
                    # + rdst - self.margin)
        return dst_loss + self.reg(x1) + self.reg(x2) + self.reg(x3)

    def disJointLoss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        d = self.classEmbeddingDict(input[:, 1])

        rc = torch.reshape(torch.abs(c[:, -1]), [-1, 1])
        rd = torch.reshape(torch.abs(d[:, -1]), [-1, 1])

        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]

        sr = rc + rd

        dst = torch.reshape(torch.linalg.norm(x2 - x1, axis=1), [-1, 1])
        relu = torch.nn.ReLU()
        return relu(sr - dst + self.margin) + self.reg(x1) + self.reg(x2)

    # cClass isSubSet of relation some dClass
    def nf3Loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]

        rc = torch.abs(c[:, -1])
        rd = torch.abs(d[:, -1])

        x3 = x1 + r
        euc = torch.linalg.norm(x3 - x2, axis=1)
        relu = torch.nn.ReLU()
        dst = torch.reshape(relu(euc + rc - rd - self.margin), [-1, 1])

        return dst + self.reg(x1) + self.reg(x2)
        # cClass is_NOT_SubSet of relation some dClass

    def neg_loss(self, input):
        c = self.classEmbeddingDict(input[:, 0])
        r = self.relationEmbeddingDict(input[:, 1])
        d = self.classEmbeddingDict(input[:, 2])

        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]
        rc = torch.abs(c[:, -1])
        rd = torch.abs(d[:, -1])

        x3 = x1 + r
        euc = torch.linalg.norm(x3 - x2, axis=1)

        #   relu = torch.nn
        dst = torch.reshape((-(euc - rc - rd) + self.margin), [-1, 1])
        return dst + self.reg(x1) + self.reg(x2)

    # relation some cClass isSubSet of dClass
    def nf4Loss(self, input):
        c = self.classEmbeddingDict(input[:, 1])

        r = self.relationEmbeddingDict(input[:, 0])

        d = self.classEmbeddingDict(input[:, 2])

        rc = torch.reshape(torch.abs(c[:, -1]), [-1, 1])
        rd = torch.reshape(torch.abs(d[:, -1]), [-1, 1])

        x1 = c[:, 0:-1]
        x2 = d[:, 0:-1]

        sr = rd - rc # todo: changed from rc + rd

        # c - r should intersect with d
        x3 = x1 - r
        dst = torch.reshape(torch.linalg.norm(x3 - x2, axis=1), [-1, 1])

        relu = torch.nn.ReLU()
        dst_loss = relu(dst - sr - self.margin)
        return dst_loss + self.reg(x1) + self.reg(x2)

    def top_loss(self, input):
        d = self.classEmbeddingDict(input[0])
        rd = torch.reshape(torch.abs(d[-1]), [-1, 1])
        return torch.abs(rd - self.inf)

    def forward(self, input):
        batch = 512

        if len(input['nf1']) == 0:
            loss1 = torch.zeros(1)
        else:
            rand_index = np.random.choice(len(input['nf1']), size=batch)
            # print(len(input['nf1']))
            nf1Data = input['nf1'][rand_index]
            nf1Data = nf1Data.to(self.device)
            loss1 = self.nf1Loss(nf1Data)

        # nf2
        if len(input['nf2']) == 0:
            loss2 = torch.zeros(1)
        else:
            rand_index = np.random.choice(len(input['nf2']), size=batch)
            #   print(input['nf2'])
            nf2Data = input['nf2'][rand_index]
            nf2Data = nf2Data.to(self.device)
            loss2 = self.nf2Loss(nf2Data)

        # nf3
        if len(input['nf3']) == 0:
            loss3 = torch.zeros(1)
        else:
            rand_index = np.random.choice(len(input['nf3']), size=batch)
            nf3Data = input['nf3'][rand_index]
            nf3Data = nf3Data.to(self.device)
            loss3 = self.nf3Loss(nf3Data)

        # nf4
        if len(input['nf4']) == 0:
            loss4 = torch.zeros(1)
        else:
            rand_index = np.random.choice(len(input['nf4']), size=batch)
            nf4Data = input['nf4'][rand_index]
            nf4Data = nf4Data.to(self.device)
            loss4 = self.nf4Loss(nf4Data)


        # disJoint
        if len(input['disjoint']) == 0:
            disJointLoss = torch.zeros(1)
        else:
            rand_index = np.random.choice(len(input['disjoint']), size=batch)

            disJointData = input['disjoint'][rand_index]
            disJointData = disJointData.to(self.device)
            disJointLoss = self.disJointLoss(disJointData)

        # negLoss
        if len(input['nf3_neg0']) == 0:
            negLoss = torch.zeros(1)
        else:
            rand_index = np.random.choice(len(input['nf3_neg0']), size=batch)
            negData = input['nf3_neg0'][rand_index]
            negData = negData.to(self.device)
            negLoss = self.neg_loss(negData)

        totalLoss = [loss1.mean() + loss2.mean() + disJointLoss.mean() + loss4.mean() + loss3.mean() + negLoss.mean()]
        return sum(totalLoss)

    def to_loaded_model(self):
        model = ElbeLoadedModel()
        model.embedding_size = self.embedding_dim
        model.class_embeds = self.classEmbeddingDict.weight.detach()
        model.relation_embeds = self.relationEmbeddingDict.weight.detach()
        return model

    def save(self, folder, best=False):
        if not os.path.exists(folder):
            os.makedirs(folder)
        suffix = '_best' if best else ''
        np.save(f'{folder}/class_embeds{suffix}.npy', self.classEmbeddingDict.weight.detach().cpu().numpy())
        np.save(f'{folder}/relations{suffix}.npy', self.relationEmbeddingDict.weight.detach().cpu().numpy())
