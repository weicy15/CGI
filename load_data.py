import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from collections import defaultdict
import numpy as np
import pandas as pd
from random import choice

import os
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



class Data(object):
    def __init__(self, interact_train, interact_test, social, user_num, item_num):
        self.interact_train = interact_train
        self.interact_test = interact_test
        self.social = social
        self.user_num = user_num
        self.item_num = item_num

        self.user_list = list(range(self.user_num))
        self.item_list = list(range(self.item_num))

        self.userMeans = {} #mean values of users's ratings
        self.itemMeans = {} #mean values of items's ratings
        self.globalMean = 0

        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self.testSet_i = defaultdict(dict)

        self.__generateSet()
        self.__computeItemMean()
        self.__computeUserMean()
        self.__globalAverage()

        self.train_dataset = Train_dataset(self.interact_train, self.item_num, self.trainSet_u)

    def __generateSet(self):
        for row in self.interact_train.itertuples(index=False):
            userName = row.userid
            itemName = row.itemid
            rating = row.score
            self.trainSet_u[userName][itemName] = rating
            self.trainSet_i[itemName][userName] = rating


        for row in self.interact_test.itertuples(index=False):
            userName = row.userid
            itemName = row.itemid
            rating = row.score
            self.testSet_u[userName][itemName] = rating
            self.testSet_i[itemName][userName] = rating


    def __globalAverage(self):
        total = sum(self.userMeans.values())
        if total==0:
            self.globalMean = 0
        else:
            self.globalMean = total/len(self.userMeans)

    def __computeUserMean(self):
        for u in self.user_list:
            self.userMeans[u] = sum(self.trainSet_u[u].values())/(len(self.trainSet_u[u]) + 0.00000001)

    def __computeItemMean(self):
        for c in self.item_list:
            self.itemMeans[c] = sum(self.trainSet_i[c].values())/(len(self.trainSet_i[c])+0.00000001)

    def userRated(self,u):
        return list(self.trainSet_u[u].keys()),list(self.trainSet_u[u].values())

    def itemRated(self,i):
        return list(self.trainSet_i[i].keys()),list(self.trainSet_i[i].values())





class Train_dataset(Dataset):
    def __init__(self, interact_train, item_num, trainSet_u):
        super(Train_dataset, self).__init__()
        self.interact_train = interact_train
        self.item_list = list(range(item_num))
        self.trainSet_u = trainSet_u

    def __len__(self):
        return len(self.interact_train)

    def __getitem__(self, idx):
        entry = self.interact_train.iloc[idx]

        # user, item, negitem
        user = entry.userid
        pos_item = entry.itemid
        neg_item = choice(self.item_list)
        while neg_item in self.trainSet_u[user]:
            neg_item = choice(self.item_list)

        return user, pos_item, neg_item

