import numpy as np
import torch


def rnet_input(trajectory: list, player: int) -> torch.Tensor:
    """Stacked observations of four previous states along with player
    indicating plane."""
    # Pad with 0
    if len(trajectory) < 4:
        pad = np.zeros(18 * (4 - len(trajectory)))
    else:
        pad = np.array([])

    inp = np.array([])
    for state in trajectory[-4:]:
        w = state == 1
        b = state == -1
        w_b = np.concatenate((w, b), axis=0).flatten()
        inp = np.append(inp, w_b)

    p = np.full_like(np.arange(9), player) == 1
    inp = np.concatenate((pad, inp, p), axis=0)

    return torch.FloatTensor(inp)


def dnet_input(hidden_state: torch.Tensor, action: int):
    a = torch.FloatTensor([0] * 9)
    a[action] = 1
    inp = torch.concat([hidden_state, a])
    return inp


if __name__ == "__main__":
    from game import draw_board
    trajectory = [
        np.array([0, 0, -1, 0, -1, 0, 1, 1, 0]),
        np.array([0, 0, -1, 0, -1, 0, 1, 1, 1])
    ]

    draw_board(trajectory[-1])
    inp_r = rnet_input(trajectory, 1)
    print(inp_r)
    hs = torch.randn(size=(16,))
    inp_d = dnet_input(hs, 4)
    print(inp_d)
