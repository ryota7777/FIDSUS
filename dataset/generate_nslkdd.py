import csv
import numpy as np
import os
import sys
import random

import pandas as pd
from pandas import read_csv
# from torchtext.data import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(10)
np.random.seed(1)
# num_clients = 100
max_len = 50
dir_path = "NSLKDD/"




# Allocate data to users
def generate_KDD(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    train = pd.read_csv(dir_path + "rawdata/datasets/KDDTrain5.csv")
    train = np.array(train)
    train_data = train[:, :-1]  # 选择第 1 和第 2 列作为数据
    train_label = train[:, -1]  # 选择最后一列作为标签

    test = pd.read_csv(dir_path + "rawdata/datasets/KDDTest5.csv")
    test = np.array(test)
    test_data = test[:, :-1]  # 选择第 1 和第 2 列作为数据
    test_label = test[:, -1]  # 选择最后一列作为标签
    train_label = train_label -1
    test_label = test_label - 1
    dataset_data = []
    dataset_label = []

    dataset_data.extend(train_data)
    dataset_data.extend(test_data)
    dataset_label.extend(train_label)
    dataset_label.extend(test_label)
    dataset_data_np = np.array(dataset_data)
    dataset_label_np = np.array(dataset_label)
    num_classes = len(set(dataset_label))
    # 计算每列的均值和标准差
    mean_vals = np.mean(dataset_data_np, axis=0)
    dataset_data_np = dataset_data_np.astype(float)
    std_vals = np.std(dataset_data_np, axis=0)

    num_zero_std = np.sum(std_vals == 0)
    std_vals[std_vals == 0] = 1.0  # 将零标准差的值设置为1.0，避免除以零

    # 对每列进行 Z-score 归一化处理
    normalized_dataset_data = (dataset_data_np - mean_vals) / std_vals

    X, y, statistic = separate_data((normalized_dataset_data, dataset_label_np), num_clients, num_classes, niid, balance, partition, class_per_client=3)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)
    # # #
    # # #


if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # balance = True if sys.argv[2] == "balance" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_KDD(dir_path, 50, niid=True , balance=False, partition="dir")