import numpy as np
import torch


def rnet_input(state: np.ndarray, player: int) -> torch.Tensor:
    w = state == 1
    b = state == -1
    p = [player] * 3
    inp = np.concatenate((w, b, p), axis=0)
    return torch.FloatTensor(inp)


def dnet_input(hidden_state: torch.Tensor, action: int):
    a = torch.FloatTensor([0] * 9)
    a[action] = 1
    inp = torch.concat([hidden_state, a])
    return inp


if __name__ == "__main__":
    from game import draw_board
    state = np.array([-1, 0, -1, 0, -1, 0, 1, 1, 0])
    draw_board(state)
    inp_r = rnet_input(state, -1)
    print(inp_r)
    hs = torch.randn(size=(9,))
    inp_d = dnet_input(hs, 4)
    print(inp_d)
