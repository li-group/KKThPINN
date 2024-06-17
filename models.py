import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

device = "cuda" if torch.cuda.is_available() else "cpu"


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim):
        super(NN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        z0 = self.layers[-1](x)
        return z0

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class NNOPT(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A, B, b):
        super(NNOPT, self).__init__()
        self.A = A
        self.B = B
        self.b = b
        self.chunk = torch.mm(B.t(),
                             torch.inverse(
                                 torch.mm(B, B.t())
                             )
                             )
        self.Astar = - torch.mm(self.chunk, self.A)
        self.Bstar = torch.eye(z0_dim).to(device) - torch.mm(self.chunk, self.B)
        self.bstar = torch.matmul(self.chunk, self.b).squeeze(-1)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, z0_dim))

        self.fc_fixed1 = nn.Linear(z0_dim, z0_dim, bias=False)  ########
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, z0_dim, bias=False)  ########
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)

    def forward(self, x):
        x0 = x
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        z0 = self.layers[-1](x)
        z = self.fc_fixed1(z0) + self.fc_fixed2(x0)
        return z

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()


class ECNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A, B_indep, B_dep, b):
        super(ECNN, self).__init__()
        self.A = A
        self.B_indep = B_indep
        self.B_dep = B_dep
        self.b = b

        self.B_dep_inverse = torch.inverse(B_dep)
        self.Astar = - torch.mm(self.B_dep_inverse, self.A)
        self.Bstar = - torch.mm(self.B_dep_inverse, self.B_indep)
        self.bstar = torch.matmul(self.B_dep_inverse, b).squeeze(-1)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(hidden_num - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, self.B_indep.shape[1]))

        self.fc_fixed1 = nn.Linear(self.B_indep.shape[1], self.B_dep.shape[1], bias=False)  ########
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, self.B_dep.shape[1], bias=False)  ########
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)
        self.fc_fixed2.bias = nn.Parameter(self.bstar, requires_grad=False)

    def forward(self, x):
        x0 = x
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        z_indep = self.layers[-1](x)
        z_dep = self.fc_fixed1(z_indep) + self.fc_fixed2(x0)
        z = torch.cat((z_dep, z_indep), dim=1)
        return z

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()