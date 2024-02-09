import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

device = "cuda" if torch.cuda.is_available() else "cpu"

# min 1/2 (z - z0)^T (z - z0)
# s.t. Ax + Bz = b
# which can be transformed into a set of equations G(z, lam) by KKT conditions:
#               | z - z0 + B^T lam |
# G(z*, lam*) = |                  | = 0
#               | Ax + Bz - b      |
# i.e.
# | I   B^T |   |  z  |    |  z0  |
# |         |   |     |  = |      |
# | B    0  |   | lam |  = | b-Ax |
# by Schur complement and block matrix inversion,
# we have closed-form solution
# z* = A*x + B*z0 + b*
# where BBB_ = B^T (B B^T)^{-1}
# A* = - BBB_ A
# B* = I - BBB_ B
# b* = BBB_ b


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim):
        super(NN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, z0_dim)
        hid_modules = []
        for _ in range(hidden_num):
            hid_modules.append(nn.ReLU())
            hid_modules.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_layers = nn.Sequential(*hid_modules)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(self.hidden_layers(out))
        z0 = self.output_layer(out)
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
        self.BBB_ = torch.mm(B.t(),
                             torch.inverse(
                                 torch.mm(B, B.t())
                             )
                             )
        self.Astar = - torch.mm(self.BBB_, self.A)
        self.Bstar = torch.eye(z0_dim).to(device) - torch.mm(self.BBB_, self.B)
        self.bstar = nn.Parameter(torch.matmul(self.BBB_, self.b), requires_grad=False)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, z0_dim)
        hid_modules = []
        for _ in range(hidden_num):
            hid_modules.append(nn.ReLU())
            hid_modules.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden_layers = nn.Sequential(*hid_modules)
        self.relu = nn.ReLU()

        self.fc_fixed1 = nn.Linear(z0_dim, z0_dim, bias=False)  ########
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, z0_dim, bias=False)  ########
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(self.hidden_layers(out))
        z0 = self.output_layer(out)
        z = self.fc_fixed1(z0) + self.fc_fixed2(x) + self.bstar
        return z

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()




class NNsimple(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim):
        super(NNsimple, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, z0_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        z0 = self.output_layer(out)
        return z0

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()



class NNOPTsimple(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, z0_dim, A, B, b):
        super(NNOPTsimple, self).__init__()
        self.A = A
        self.B = B
        self.b = b
        self.BBB_ = torch.mm(B.t(),
                             torch.inverse(
                                 torch.mm(B, B.t())
                             )
                             )
        self.Astar = - torch.mm(self.BBB_, self.A)
        self.Bstar = torch.eye(z0_dim).to(device) - torch.mm(self.BBB_, self.B)
        self.bstar = nn.Parameter(torch.matmul(self.BBB_, self.b), requires_grad=False)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, z0_dim)
        self.relu = nn.ReLU()

        self.fc_fixed1 = nn.Linear(z0_dim, z0_dim, bias=False)  ########
        self.fc_fixed1.weight = nn.Parameter(self.Bstar, requires_grad=False)
        self.fc_fixed2 = nn.Linear(input_dim, z0_dim, bias=False)  ########
        self.fc_fixed2.weight = nn.Parameter(self.Astar, requires_grad=False)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.relu(out)
        z0 = self.output_layer(out)
        z = self.fc_fixed1(z0) + self.fc_fixed2(x) + self.bstar
        return z

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.fc_fixed1 and module is not self.fc_fixed2:
                module.reset_parameters()
