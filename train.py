import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import numpy as np
from collections import defaultdict

from models import Model
import argparse

import load_data

import os
import time
import shutil
from tqdm import tqdm

cuda_device = '3'

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=str, default='')
    parser.add_argument("--dataset_name", type=str, default='douban')
    parser.add_argument("--social_data", type=bool, default=False)
    # test_set/cv/split
    parser.add_argument("--load_mode", type=str, default='test_set')

    parser.add_argument("--implcit_bottom", type=int, default=3)
    parser.add_argument("--cross_validate", type=int, default=None)
    parser.add_argument("--split", type=float, default=None)
    parser.add_argument("--user_fre_threshold", type=int, default=None)
    parser.add_argument("--item_fre_threshold", type=int, default=None)

    parser.add_argument("--loadFilename", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=1000)

    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--L", type=int, default=3)

    parser.add_argument("--ssl_temp", type=float, default=0.2)
    parser.add_argument("--choosing_tmp", type=float, default=0.2)
    parser.add_argument("--rec_loss_reg", type=float, default=1)
    parser.add_argument("--ssl_loss_reg", type=float, default=0.02)
    parser.add_argument("--walk_length", type=int, default=10)


    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--lr_decay_every_step", type=int, default=5)

    parser.add_argument("--K_list", type=int, nargs='+', default=[10, 20, 50])

    opt = parser.parse_args()
    return opt

def my_collate_train(batch):
    user_id = [item[0] for item in batch]
    pos_item = [item[1] for item in batch]
    neg_item = [item[2] for item in batch]

    user_id = torch.LongTensor(user_id)
    pos_item = torch.LongTensor(pos_item)
    neg_item = torch.LongTensor(neg_item)

    return [user_id, pos_item, neg_item]


def one_train(Data, opt):
    print(opt)
    print('Building dataloader >>>>>>>>>>>>>>>>>>>')

    train_dataset = Data.train_dataset
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=opt.batch_size, collate_fn=my_collate_train)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:{0}".format(cuda_device))

    print(device)
    if opt.loadFilename != None:
        checkpoint = torch.load(opt.loadFilename)
        sd = checkpoint['sd']
        optimizer_sd = checkpoint['opt']

    print("building model >>>>>>>>>>>>>>>")
    model = Model(Data, opt, device)

    if opt.loadFilename != None:
        model.load_state_dict(sd)


    print('Building optimizers >>>>>>>')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_every_step, gamma=opt.lr_decay)


    print('Start training...')
    start_epoch = 0
    if opt.loadFilename != None:
        checkpoint = torch.load(opt.loadFilename)
        start_epoch = checkpoint['epoch'] + 1

    model = model.to(device)
    model.train()

    for epoch in range(start_epoch, opt.epoch):
        with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
            for index, (user_id, pos_item, neg_item) in enumerate(train_loader):
                
                user_id = user_id.to(device)
                pos_item = pos_item.to(device)
                neg_item = neg_item.to(device)

                loss = model(user_id, pos_item, neg_item)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

        scheduler.step()

    model.eval()
    
    torch.save({
        'sd': model.state_dict(),
    }, 'model.tar')
        
opt = get_config()
Data = load_data.Data(interact_train, interact_test, social, user_num, item_num)
one_train(Data, opt)

