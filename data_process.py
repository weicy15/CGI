from posixpath import split
import numpy as np
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from collections import defaultdict

core = cpu_count()

def parallelize(data, func, num_of_processes=core):
    data_split = np.array_split(data, num_of_processes)
    with Pool(num_of_processes) as pool:
        data_list = pool.map(func, data_split)
    data = pd.concat(data_list)
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=core):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

def get_item_count(item_list, row):
    item_count = item_list.count(row.itemid)
    return item_count

def get_all_item_counts(data, item_list):
    item_counts = parallelize_on_rows(data, partial(get_item_count, deepcopy(item_list)))
    return item_counts

def get_user_count(user_list, row):
    user_count = user_list.count(row.userid)
    return user_count

def get_all_user_counts(data, user_list):
    user_counts = parallelize_on_rows(data, partial(get_user_count, deepcopy(user_list)))
    return user_counts



def get_user_keep(history_users, row):
    return (row.userid in history_users)

def get_all_user_keeps(data, history_users):
    user_keeps = parallelize_on_rows(data, partial(get_user_keep, deepcopy(history_users)))
    return user_keeps



def get_item_keep(history_items, row):
    return (row.itemid in history_items)

def get_all_item_keeps(data, history_items):
    item_keeps = parallelize_on_rows(data, partial(get_item_keep, deepcopy(history_items)))
    return item_keeps



def data_process(dataset_name, split_rate=0.9, user_fre_threshold = None, item_fre_threshold = None, socia_data= False):
    save_dir = "dataset/" + dataset_name
    if not os.path.exists(save_dir):
        print("dataset is not exist!!!!")
        return None

    social = None

    interact = pd.read_table(save_dir+"/ratings.txt", sep="\t", header= None, names= ['userid', 'itemid', 'score'])

    if user_fre_threshold != None and item_fre_threshold != None:
        item_list = interact['itemid'].tolist()
        user_list = interact['userid'].tolist()

        item_counts = get_all_item_counts(interact, item_list)
        user_counts = get_all_user_counts(interact, user_list)

        interact = interact[(item_counts > item_fre_threshold) & (user_counts > user_fre_threshold)]


    if socia_data == True:
        social = pd.read_table(save_dir+"/trusts.txt", sep='\t', header= None, names= ['src', 'dst'])
        social = social[['src', 'dst']]



    user_encoder = LabelEncoder()
    if socia_data == True:
        user_encoder.fit(pd.concat([interact['userid'], social['src'], social['dst']]))
        interact['userid'] = user_encoder.transform(interact['userid'])
        social['src'] = user_encoder.transform(social['src'])
        social['dst'] = user_encoder.transform(social['dst'])
        social.to_pickle(save_dir + "/social.pkl")
        
    else:
        user_encoder.fit(interact['userid'])
        interact['userid'] = user_encoder.transform(interact['userid'])


    item_encoder = LabelEncoder()
    interact['itemid'] = item_encoder.fit_transform(interact['itemid'])

    user_encoder_map = pd.DataFrame(
        {'encoded': range(len(user_encoder.classes_)), 'user': user_encoder.classes_})
    user_encoder_map.to_csv(save_dir + '/user_encoder_map.csv', index=False)

    item_encoder_map = pd.DataFrame(
        {'encoded': range(len(item_encoder.classes_)), 'item': item_encoder.classes_})
    item_encoder_map.to_csv(save_dir + '/item_encoder_map.csv', index=False)



    interact_train, interact_test = train_test_split(interact, train_size=split_rate, random_state=5)


    history_users = list(set(interact_train['userid'].tolist()))
    history_items = list(set(interact_train['itemid'].tolist()))

    user_keeps = get_all_user_keeps(interact_test, history_users)
    item_keeps = get_all_item_keeps(interact_test, history_items)

    interact_test = interact_test[user_keeps & item_keeps]

    interact_train.to_pickle(save_dir + "/interact_train.pkl")
    interact_test.to_pickle(save_dir + "/interact_test.pkl")



dataset_name = 'douban'
data_process(dataset_name, split_rate=0.9, user_fre_threshold = 5, item_fre_threshold = 5, socia_data= True)