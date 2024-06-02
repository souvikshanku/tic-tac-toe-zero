import torch
import torch.nn as nn
import torch.nn.functional as F


class ReprNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.inp_dim = 81
        self.hs_dim = 16

        self.fc1 = nn.Linear(self.inp_dim, 64)
        self.fc2 = nn.Linear(64, self.hs_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

    def predict(self, x):
        with torch.no_grad():
            hidden_state = self.forward(x)

        return hidden_state


class DynmNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.inp_dim = 16 + 9
        self.hs_dim = 16

        self.fc1 = nn.Linear(self.inp_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.hs_dim)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    def predict(self, x):
        with torch.no_grad():
            next_state = self.forward(x)

        return next_state


class PredNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.inp_dim = 16
        self.action_size = 9

        self.fc1 = nn.Linear(self.inp_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.fc3 = nn.Linear(64, self.action_size)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, self.inp_dim)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        pi = self.fc3(x)
        v = self.fc4(x)

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, hidden_state):
        with torch.no_grad():
            policy, value = self.forward(hidden_state)

        return policy, value
