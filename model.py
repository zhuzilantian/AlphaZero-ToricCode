import os
import numpy as np
from typing import Tuple
import torch
from torch import nn, Tensor


class ToricNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int], output_n: int):
        super().__init__()
        input_n = int(np.prod(input_shape))
        out1 = int(2 ** np.ceil(np.log2(input_n)))
        out2 = int(out1 / 2)

        self.feature = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_n, out1),
            nn.ReLU(),
            nn.Linear(out1, out2),
            nn.ReLU()
        )
        self.output_p = nn.Sequential(
            nn.Linear(out2, output_n),
            nn.Softmax(dim=1)
        )
        self.output_v = nn.Sequential(
            nn.Linear(out2, 1),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        x = self.feature(x)
        probs = self.output_p(x)
        value = self.output_v(x)
        return probs, value

    def save(self, path):
        dirs, _ = os.path.split(path)
        if dirs != '':
            os.makedirs(dirs, exist_ok=True)

        try:
            torch.save(self.state_dict(), path)
        except OSError as e:
            print(e.strerror)

    def load(self, path):
        try:
            self.load_state_dict(torch.load(path))
        except FileNotFoundError as e:
            print(f"{e.strerror}: '{e.filename}'")
        except OSError as e:
            print(e.strerror)


class AlphaZeroLoss(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c

    def forward(self, p: Tensor, p_t: Tensor, v: Tensor, v_t: Tensor) -> Tensor:
        loss_v = torch.pow(v_t - v, 2)
        loss_p = p_t * torch.log(p)
        loss_p = -loss_p.sum(dim=1, keepdim=True)
        loss = torch.mean(loss_v + self.c * loss_p)
        return loss
