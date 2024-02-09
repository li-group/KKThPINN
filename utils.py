import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data


def load_data(dataset_path):
    dataset = np.array(pd.read_csv(dataset_path).values)
    scaler = MaxAbsScaler()
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    return dataset, scaler


def get_ScaleAndMean(scaler, x_dim, z_dim):
    xscale = []
    zscale = []
    for idx in range(x_dim):
        xscale.append(scaler.scale_[idx])
    for idx in range(z_dim):
        zscale.append((scaler.scale_[idx+x_dim]))
    return xscale, zscale


def get_scaledABb(A, B, b, scaler):
    x_dim = A.shape[1]
    z_dim = B.shape[1]
    xscale, zscale = get_ScaleAndMean(scaler, x_dim, z_dim)
    xscale, zscale = torch.tensor(xscale), torch.tensor(zscale)
    A_scale = torch.ones_like(A) * xscale
    B_scale = torch.ones_like(B) * zscale
    A_scaled = A * A_scale
    B_scaled = B * B_scale
    b_scaled = b
    return A_scaled.float(), B_scaled.float(), b_scaled.float()


class Data_cstr(data.Dataset):
    def __init__(self, dataset):
        self.dataset_tensor = torch.from_numpy(dataset).float()
        self.X = self.dataset_tensor[:, :3]
        self.Y = self.dataset_tensor[:, 3:]
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)  # initial val_ratio -> 0.2

        self.A = torch.tensor([[0, 1, -1],
                                [0, 1, 0]]).float()  # (2, 3)
        self.B = torch.tensor([[0, -1, 1],
                                [-1, -1, 0]]).float()  # (2, 3)
        self.b = torch.tensor([0, 0]).float()
        #### NOTE: IT IS NOT SCALED ####

        self.constrained_indexes = list(set([index for index in torch.nonzero(self.B)[:, -1].tolist()]))
        self.unconstrained_indexes = [item for item in range(self.B.shape[1]) if item not in self.constrained_indexes]

    def __len__(self):
        return len(self.dataset_tensor)

    def __getitem__(self, idx):
        return self.dataset_tensor[idx, :]

    def split_data(self, val_ratio, test_ratio=0.2):
        XY = data.TensorDataset(self.X, self.Y)
        n_samples = len(XY)
        n_val = int(val_ratio * n_samples)
        n_test = int(test_ratio * n_samples)
        n_train = n_samples - n_val - n_test
        # train_set, val_set, test_set = data.random_split(XY, [n_train, n_val, n_test])
        train_set = data.Subset(XY, range(0, n_train))
        val_set = data.Subset(XY, range(n_train, n_train + n_val))
        test_set = data.Subset(XY, range(n_train + n_val, n_samples))
        return train_set, val_set, test_set

    def resplit_data(self, val_ratio, test_ratio=0.2):
        self.train_set, self.val_set, self.test_set = self.split_data(val_ratio, test_ratio)


class Data_plant(data.Dataset):
    def __init__(self, dataset):
        self.dataset_tensor = torch.from_numpy(dataset).float()
        self.X = self.dataset_tensor[:, :4]
        self.Y = self.dataset_tensor[:, 4:]
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)  # initial val_ratio -> 0.2

        self.A = torch.tensor([[1, 1, 1, -1]]).float()  # (1, 4)
        self.B = torch.tensor([[-1, 0, -1, 0, -1]]).float()  # (1, 5)
        self.b = torch.tensor([0]).float()  # (1, )
        #### NOTE: IT IS NOT SCALED ####

        self.constrained_indexes = list(set([index for index in torch.nonzero(self.B)[:, -1].tolist()]))
        self.unconstrained_indexes = [item for item in range(self.B.shape[1]) if item not in self.constrained_indexes]

    def __len__(self):
        return len(self.dataset_tensor)

    def __getitem__(self, idx):
        return self.dataset_tensor[idx, :]

    def split_data(self, val_ratio, test_ratio=0.2):
        XY = data.TensorDataset(self.X, self.Y)
        n_samples = len(XY)
        n_val = int(val_ratio * n_samples)
        n_test = int(test_ratio * n_samples)
        n_train = n_samples - n_val - n_test
        # train_set, val_set, test_set = data.random_split(XY, [n_train, n_val, n_test])
        train_set = data.Subset(XY, range(0, n_train))
        val_set = data.Subset(XY, range(n_train, n_train + n_val))
        test_set = data.Subset(XY, range(n_train + n_val, n_samples))
        return train_set, val_set, test_set

    def resplit_data(self, val_ratio, test_ratio=0.2):
        self.train_set, self.val_set, self.test_set = self.split_data(val_ratio, test_ratio)


class Data_distillation(data.Dataset):
    def __init__(self, dataset):
        self.dataset_tensor = torch.from_numpy(dataset).float()
        self.X = self.dataset_tensor[:, :5]
        self.Y = self.dataset_tensor[:, 5:]
        self.train_set, self.val_set, self.test_set = self.split_data(0.2)  # initial val_ratio -> 0.2

        self.A = torch.tensor([[0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1]]).float()  # (2, 5)
        self.B = torch.tensor([[-1, 0, 0, 0, 0, 0, -1, -1, 0, 0],
                                [0, -1, 0, 0, 0, 0, 0, 0, -1, -1]]).float()  # (2, 10)
        self.b = torch.tensor([0, 0]).float()  # (2, )
        #### NOTE: IT IS NOT SCALED ####

        self.constrained_indexes = list(set([index for index in torch.nonzero(self.B)[:, -1].tolist()]))
        self.unconstrained_indexes = [item for item in range(self.B.shape[1]) if item not in self.constrained_indexes]

    def __len__(self):
        return len(self.dataset_tensor)

    def __getitem__(self, idx):
        return self.dataset_tensor[idx, :]

    def split_data(self, val_ratio, test_ratio=0.2):
        XY = data.TensorDataset(self.X, self.Y)
        n_samples = len(XY)
        n_val = int(val_ratio * n_samples)
        n_test = int(test_ratio * n_samples)
        n_train = n_samples - n_val - n_test
        # train_set, val_set, test_set = data.random_split(XY, [n_train, n_val, n_test])
        train_set = data.Subset(XY, range(0, n_train))
        val_set = data.Subset(XY, range(n_train, n_train + n_val))
        test_set = data.Subset(XY, range(n_train + n_val, n_samples))
        return train_set, val_set, test_set

    def resplit_data(self, val_ratio, test_ratio=0.2):
        self.train_set, self.val_set, self.test_set = self.split_data(val_ratio, test_ratio)


class PINNLoss(nn.Module):
    def __init__(self, A, B, b, eta, reduction='mean'):
        super(PINNLoss, self).__init__()
        self.A = A
        self.B = B
        self.b = b
        self.eta = eta
        self.reduction = reduction

    def forward(self, X, input, target):
        mse_loss = F.mse_loss(input, target, reduction=self.reduction)
        pinn_loss = torch.mean(self.eta * (torch.mm(self.B, input.T) + torch.mm(self.A, X.T) - self.b.unsqueeze(1))**2)
        return mse_loss + pinn_loss