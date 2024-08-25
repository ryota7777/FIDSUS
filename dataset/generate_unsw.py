import numpy as np
import os
import random
import pandas as pd
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
max_len = 50
dir_path = "UNSW/"




# Allocate data to users
def generate_unsw(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return
    train = pd.read_csv(dir_path + "rawdata/datasets/unsw_train10.csv")
    train = np.array(train)
    train_data = train[:, 1:-2]
    train_label = train[:, -1]
    test = pd.read_csv(dir_path + "rawdata/datasets/unsw_test10.csv")
    test = np.array(test)
    test_data = test[:, 1:-2]
    test_label = test[:, -1]
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
    mean_vals = np.mean(dataset_data_np, axis=0)
    dataset_data_np = dataset_data_np.astype(float)
    std_vals = np.std(dataset_data_np, axis=0)
    normalized_dataset_data = (dataset_data_np - mean_vals) / std_vals

    X, y, statistic = separate_data((normalized_dataset_data, dataset_label_np), num_clients, num_classes, niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)
if __name__ == "__main__":

    generate_unsw(dir_path, 50, niid=True , balance=False, partition="dir")