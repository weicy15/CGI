import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict

from model import Model

from util import load_data
from util import metric
from util.early_stop import EarlyStopping

import os
import time
import shutil
from tqdm import tqdm

import random
from rectorch.metrics import Metrics



def ranking_meansure(pred_scores, ground_truth, k):
    # user_num * item_num (score)
    # user_num * item_num (1/0)

    # user_num
    ndcg_list = Metrics.ndcg_at_k(pred_scores, ground_truth, k).tolist()
    recall_list = Metrics.recall_at_k(pred_scores, ground_truth, k).tolist()
    mrr_list = Metrics.mrr_at_k(pred_scores, ground_truth, k).tolist()

    return np.mean(ndcg_list), np.mean(recall_list), np.mean(mrr_list)



def my_collate_test(batch):
    user_id = [item[0] for item in batch]
    pos_item_tensor = [item[1] for item in batch]

    user_id = torch.LongTensor(user_id)
    pos_item = torch.stack(pos_item_tensor)

    return [user_id, pos_item]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


loadFilename = "model.tar"
checkpoint = torch.load(loadFilename, map_location=device)
sd = checkpoint['sd']
opt = checkpoint['opt']

interact_train, interact_test, social, user_num, item_num = load_data.data_load(opt.dataset_name, social_data=opt.social_data, test_dataset= True, bottom=opt.implcit_bottom)
Data = load_data.Data(interact_train, interact_test, social, user_num, item_num)


print('Building dataloader >>>>>>>>>>>>>>>>>>>')
test_dataset = Data.test_dataset
test_loader = DataLoader(
    test_dataset, shuffle=False, batch_size=opt.batch_size, collate_fn=my_collate_test)




print("building model >>>>>>>>>>>>>>>")
model = Model(Data, opt, device)

model.load_state_dict(sd)
model = model.to(device)
model.eval()
user_historical_mask = Data.user_historical_mask.to(device)

NDCG = defaultdict(list)
RECALL = defaultdict(list)

with tqdm(total=len(test_loader), desc="predicting") as pbar:
    for i, (user_id, pos_item) in enumerate(test_loader):
        user_id = user_id.to(device)
        score = model.predict(user_id)
        score = torch.mul(user_historical_mask[user_id], score).cpu().detach().numpy()
        ground_truth = pos_item.detach().numpy()

        for K in opt.K_list:
            ndcg, recall, mrr = metric.ranking_meansure(score, ground_truth, K)
            NDCG[K].append(ndcg)
            RECALL[K].append(recall)

        pbar.update(1)

for K in opt.K_list:
    print("NDCG@{}: {}".format(K, np.mean(NDCG[K])))
    print("RECALL@{}: {}".format(K, np.mean(RECALL[K])))