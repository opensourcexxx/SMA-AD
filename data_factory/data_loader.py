import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import json

def get_dataset_np(dataset,variable):
    workspace = "../TSAD/data"
    if "WT" in dataset:
        train_df = pd.read_csv(f'{workspace}/WT/{variable}/train_orig.csv', sep=",", header=None, dtype=np.float32).dropna(
            axis=0)
        test_df = pd.read_csv(f'{workspace}/WT/{variable}/test_orig.csv', sep=",", header=None, dtype=np.float32).dropna(
            axis=0)
        dim = 10
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()
    elif "SMD" in dataset:
        train_df = pd.read_csv(
            f'{workspace}/SMD/train/machine-{variable}.txt', header=None, sep=",", dtype=np.float32)
        test_df = pd.read_csv(
            f'{workspace}/SMD/test/machine-{variable}.txt', header=None, sep=",", dtype=np.float32)

        # Get test anomaly labels
        test_labels = np.genfromtxt(
            f'{workspace}/SMD/test_label/machine-{variable}.txt', dtype=np.float32, delimiter=',')
        return train_df.to_numpy(), test_df.to_numpy(), test_labels
    elif "SMAP" in dataset:
        train = np.load(f'{workspace}/SMAP/train/{variable}.npy')
        test = np.load(f'{workspace}/SMAP/test/{variable}.npy')
        test_label = np.zeros(len(test), dtype=np.float32)

        # Set test anomaly labels from files
        labels = pd.read_csv(
            f'{workspace}/SMAP/labeled_anomalies.csv', sep=",", index_col="chan_id")
        label_str = labels.loc[variable, "anomaly_sequences"]
        label_list = json.loads(label_str)
        for i in label_list:
            test_label[i[0]:i[1] + 1] = 1.0
        return train, test, test_label
    elif "MSL" in dataset:
        train = np.load(f'{workspace}/SMAP/train/{variable}.npy')
        test = np.load(f'{workspace}/SMAP/test/{variable}.npy')
        test_label = np.zeros(len(test), dtype=np.float32)

        # Set test anomaly labels from files
        labels = pd.read_csv(
            f'{workspace}/SMAP/labeled_anomalies.csv', sep=",", index_col="chan_id")
        label_str = labels.loc[variable, "anomaly_sequences"]
        label_list = json.loads(label_str)
        for i in label_list:
            test_label[i[0]:i[1] + 1] = 1.0
        return train, test, test_label
    elif "WADI" in dataset:
        if variable == "A2":
            dim = 82
        elif variable == "A1":
            dim = 127
        train_df = pd.read_csv(f'{workspace}/WADI/{variable}/train.csv', sep=",", header=None, skiprows=1,
                               dtype=np.float32).fillna(0)
        test_df = pd.read_csv(f'{workspace}/WADI/{variable}/test.csv', sep=",", header=None, skiprows=1,
                              dtype=np.float32).fillna(0)
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()
    elif "SWAT" in dataset:
        dim = 0
        if variable == "A1_A2":
            dim = 50
        elif variable == "A4_A5":
            dim = 77
        train_df = pd.read_csv(f'{workspace}/SWAT/{variable}/train.csv', sep=",", header=None, skiprows=1,
                               dtype=np.float32).fillna(0)
        test_df = pd.read_csv(f'{workspace}/SWAT/{variable}/test.csv', sep=",", header=None, skiprows=1,
                              dtype=np.float32).fillna(0)
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        train_df.drop(dim, axis=1, inplace=True)
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()
    elif "BATADAL" in dataset:
        dim = 43
        train_df = pd.read_csv(f'{workspace}/BATADAL/train.csv', sep=",", header=None, skiprows=1,
                               dtype=np.float32).fillna(0)
        test_df = pd.read_csv(f'{workspace}/BATADAL/test.csv', sep=",", header=None, skiprows=1,
                              dtype=np.float32).fillna(0)
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)

        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        train_df.drop(dim, axis=1, inplace=True)
        test_df.drop(dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()
    elif "HADES" in dataset:
        name = "hades"
        train = np.load(f'{workspace}/data_for_tsad/{name}_kpi_train.npy')
        test = np.load(f'{workspace}/data_for_tsad/{name}_kpi_test.npy')
        test_label = np.load(f'{workspace}/data_for_tsad/label_{name}_kpi_test.npy')
        
        return train, test, test_label
    elif "YZH" in dataset:
        name = "yzh"
        train = np.load(f'{workspace}/data_for_tsad/{name}_kpi_train.npy')
        test = np.load(f'{workspace}/data_for_tsad/{name}_kpi_test.npy')
        test_label = np.load(f'{workspace}/data_for_tsad/label_{name}_kpi_test.npy')
        
        return train, test, test_label
    elif "ZTE" in dataset:
        name = "zte"
        train = np.load(f'{workspace}/data_for_tsad/{name}_kpi_train.npy')
        test = np.load(f'{workspace}/data_for_tsad/{name}_kpi_test.npy')
        test_label = np.load(f'{workspace}/data_for_tsad/label_{name}_kpi_test.npy')
        
        return train, test, test_label
    elif "ZX" in dataset:
        # dim = 13
        train_df = pd.read_csv(
            f'{workspace}/ZX/train/{variable}.csv', sep=",", header=None, skiprows=1).fillna(0)
        test_df = pd.read_csv(
            f'{workspace}/ZX/test/{variable}.csv', sep=",", header=None, skiprows=1).fillna(0)
        aux_dim = len(list(train_df)) - 1
        train_df.drop(0, axis=1, inplace=True)
        test_df.drop(0, axis=1, inplace=True)
        dim = len(list(train_df)) - 1
        # train_df["y"] = np.zeros(train_df.shape[0], dtype=np.float32)
        # Get test anomaly labels
        test_label = test_df.iloc[:, dim]
        train_df.drop(aux_dim, axis=1, inplace=True)
        test_df.drop(aux_dim, axis=1, inplace=True)
        return train_df.to_numpy(), test_df.to_numpy(), test_label.to_numpy()

def mask_data(data,removed_dims):
    data2 = data.copy()
    for i in removed_dims: # mask
        data2[:,i] = np.zeros_like(data2[:,i])
    return data2


class PSMSegLoader(object):
    def __init__(self,removed_dims, random_mask_rate, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.selected_dims = []
        self.removed_dims = removed_dims
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        
        self.random_mask_rate = random_mask_rate
        _ ,self.c = data.shape
        if len(removed_dims) > 0:
            data = mask_data(data,self.removed_dims) # mask
            self.test = mask_data(self.test,self.removed_dims) # mask

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        mask_rate = self.random_mask_rate
        mask_num = int(mask_rate*self.c)
        mask_tabel = np.random.choice(self.c,mask_num,replace=False)
        if self.mode == "train":
            out_data = np.float32(self.train[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'val'):
            out_data = np.float32(self.val[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'test'):
            out_data = np.float32(self.test[index:index + self.win_size])
            out_label = np.float32(self.test_labels[index:index + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        else:
            out_data = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_label = np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        return out_data, out_label, out_masked_data


class MSLSegLoader(object):
    def __init__(self,removed_dims,random_mask_rate,  data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.selected_dims = []
        self.removed_dims = removed_dims
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)
        
        self.random_mask_rate = random_mask_rate
        _ ,self.c = data.shape
        if len(removed_dims) > 0:
            data = mask_data(data,self.removed_dims) # mask
            self.test = mask_data(self.test,self.removed_dims) # mask

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        mask_rate = self.random_mask_rate
        mask_num = int(mask_rate*self.c)
        mask_tabel = np.random.choice(self.c,mask_num,replace=False)
        if self.mode == "train":
            out_data = np.float32(self.train[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'val'):
            out_data = np.float32(self.val[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'test'):
            out_data = np.float32(self.test[index:index + self.win_size])
            out_label = np.float32(self.test_labels[index:index + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        else:
            out_data = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_label = np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        return out_data, out_label, out_masked_data


class SMAPSegLoader(object):
    def __init__(self,removed_dims,random_mask_rate, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.selected_dims = []
        self.removed_dims = removed_dims
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)
        
        self.random_mask_rate = random_mask_rate
        _ ,self.c = data.shape
        if len(removed_dims) > 0:
            data = mask_data(data,self.removed_dims) # mask
            self.test = mask_data(self.test,self.removed_dims) # mask

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        mask_rate = self.random_mask_rate
        mask_num = int(mask_rate*self.c)
        mask_tabel = np.random.choice(self.c,mask_num,replace=False)
        if self.mode == "train":
            out_data = np.float32(self.train[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'val'):
            out_data = np.float32(self.val[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'test'):
            out_data = np.float32(self.test[index:index + self.win_size])
            out_label = np.float32(self.test_labels[index:index + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        else:
            out_data = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_label = np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        return out_data, out_label, out_masked_data


class SMDSegLoader(object):
    def __init__(self,removed_dims, random_mask_rate, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.selected_dims = []
        self.removed_dims = removed_dims
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        
        self.random_mask_rate = random_mask_rate
        _ ,self.c = data.shape
        if len(removed_dims) > 0:
            data = mask_data(data,self.removed_dims) # mask
            self.test = mask_data(self.test,self.removed_dims) # mask
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        mask_rate = self.random_mask_rate
        mask_num = int(mask_rate*self.c)
        mask_tabel = np.random.choice(self.c,mask_num,replace=False)
        if self.mode == "train":
            out_data = np.float32(self.train[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'val'):
            out_data = np.float32(self.val[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'test'):
            out_data = np.float32(self.test[index:index + self.win_size])
            out_label = np.float32(self.test_labels[index:index + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        else:
            out_data = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_label = np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        return out_data, out_label, out_masked_data

class SegLoader(object):
    def __init__(self,dataset,removed_dims, random_mask_rate, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.selected_dims = []
        self.removed_dims = removed_dims
        
        dataset, group = dataset.split(" ")
        data, test_data, test_labels = get_dataset_np(dataset,group)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        
        self.random_mask_rate = random_mask_rate
        _ ,self.c = data.shape
        if len(removed_dims) > 0:
            data = mask_data(data,self.removed_dims) # mask
            self.test = mask_data(self.test,self.removed_dims) # mask
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_labels

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        mask_rate = self.random_mask_rate
        mask_num = int(mask_rate*self.c)
        mask_tabel = np.random.choice(self.c,mask_num,replace=False)
        if self.mode == "train":
            out_data = np.float32(self.train[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'val'):
            out_data = np.float32(self.val[index:index + self.win_size])
            out_label = np.float32(self.test_labels[0:self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        elif (self.mode == 'test'):
            out_data = np.float32(self.test[index:index + self.win_size])
            out_label = np.float32(self.test_labels[index:index + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        else:
            out_data = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_label = np.float32(self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
            out_masked_data = mask_data(out_data,mask_tabel)
        return out_data, out_label, out_masked_data

def get_loader_segment(data_path, batch_size, win_size=100, step=1, mode='train', dataset='KDD', random_mask_rate = 0,config=None):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(config["removed_dims"], random_mask_rate,data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(config["removed_dims"], random_mask_rate,data_path, win_size, step, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(config["removed_dims"], random_mask_rate,data_path, win_size, step, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(config["removed_dims"], random_mask_rate,data_path, win_size, step, mode)
    else:
        dataset = SegLoader(dataset,config["removed_dims"], random_mask_rate, data_path, win_size, step, mode)
    
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
