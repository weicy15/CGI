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
        self.test_dataset = Test_dataset(self.testSet_u, self.item_num)

        user_historical_mask = np.ones((user_num, item_num))
        for uuu in self.trainSet_u.keys():
            item_list = list(self.trainSet_u[uuu].keys())
            if len(item_list) != 0:
                user_historical_mask[uuu, item_list] = 0
        

        self.user_historical_mask = torch.from_numpy(user_historical_mask)

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


class Test_dataset(Dataset):
    def __init__(self, testSet_u, item_num):
        super(Test_dataset, self).__init__()

        self.testSet_u = testSet_u
        self.user_list = list(testSet_u.keys())
        self.item_num = item_num

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item_list = torch.tensor(list(self.testSet_u[user].keys()))
        tensor = torch.zeros(self.item_num).scatter(0, item_list, 1)
        return user, tensor


def data_load(dataset_name, social_data= False, test_dataset= True, bottom=0, cv =None, split=None, user_fre_threshold = None, item_fre_threshold = None):
    save_dir = "dataset/" + dataset_name
    if not os.path.exists(save_dir):
        print("dataset is not exist!!!!")
        return None

    social = None

    if social_data == True:
        social = pd.read_table(save_dir+"/trusts.txt", sep='\t', header= None, names= ['src', 'dst'])
        social = social[['src', 'dst']]

    if test_dataset == True:
        interact_train = pd.read_pickle(save_dir + '/interact_train.pkl')
        interact_test = pd.read_pickle(save_dir + '/interact_test.pkl')
        if social_data == True:
            social = pd.read_pickle(save_dir + '/social.pkl')
        item_encoder_map = pd.read_csv(save_dir + '/item_encoder_map.csv')
        item_num = len(item_encoder_map)
        user_encoder_map = pd.read_csv(save_dir + '/user_encoder_map.csv')
        user_num = len(user_encoder_map)

        if bottom != None:
            interact_train = interact_train[interact_train['score'] > bottom]
            interact_test = interact_test[interact_test['score'] > bottom]

        return interact_train, interact_test, social, user_num, item_num


    interact = pd.read_table(save_dir+"/ratings.txt", sep='\t', header= None, names= ['userid', 'itemid', 'score'])

    if user_fre_threshold != None and item_fre_threshold != None:
        item_list = interact['itemid'].tolist()
        user_list = interact['userid'].tolist()

        item_counts = get_all_item_counts(interact, item_list)
        user_counts = get_all_user_counts(interact, user_list)

        interact = interact[(item_counts > item_fre_threshold) and (user_counts > user_fre_threshold)]


    # label encoder
    # encode IDs
    user_encoder = LabelEncoder()
    if social_data== True:
        user_encoder.fit(pd.concat([interact['userid'],social['src'],social['dst']]))
        interact['userid'] = user_encoder.transform(interact['userid'])
        social['src'] = user_encoder.transform(social['src'])
        social['dst'] = user_encoder.transform(social['dst'])
    else:
        user_encoder.fit(interact['userid'])
        interact['userid'] = user_encoder.transform(interact['userid'])

    item_encoder = LabelEncoder()
    interact['itemid'] = item_encoder.fit_transform(interact['itemid'])

    user_num = len(user_encoder.classes_)
    item_num = len(item_encoder.classes_)

    if bottom != None:
        interact = interact[interact['score'] > bottom]


    if cv != None:
        kf = KFold(n_splits=cv)
        split_iterator = kf.split(X)
        return interact, split_iterator, social, user_num, item_num

    if split != None:
        interact_train, interact_test = train_test_split(interact, train_size=split, random_state=5)
        return interact_train, interact_test, social, user_num, item_num