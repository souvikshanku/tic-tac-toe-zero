import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
# from matplotlib import pyplot as plt

from models import ReprNet, PredNet, DynmNet
from replay_buffer import generate_replay_buffer, make_targets
from utils import rnet_input, dnet_input


def get_rnet_batch_input(rb: list) -> torch.Tensor:
    batch_inp_r = []

    for game in rb:
        inp = rnet_input(game[0][0], game[0][1])
        batch_inp_r.append(inp)

    batch_inp_r = torch.cat(batch_inp_r).view(-1, batch_inp_r[0].shape[0])
    return batch_inp_r


def get_dnet_batch_input(
    hidden_states: torch.Tensor,
    rb: list,
    step: int
) -> torch.Tensor:
    batch_inp_d = []
    for i, game in enumerate(rb):
        hs = hidden_states[i]
        action = game[step][3]
        inp = dnet_input(hs, action)
        batch_inp_d.append(inp)

    batch_inp_d = torch.cat(batch_inp_d).view(-1, batch_inp_d[0].shape[0])
    return batch_inp_d


def get_target_policy_value(
    rb: list,
    step: int
) -> tuple[torch.Tensor, torch.Tensor]:
    policy = np.array([game[step][2] for game in rb])
    policy = torch.FloatTensor(policy).view(-1, policy[0].shape[0])
    value = np.array([game[step][-1] for game in rb])
    value = torch.FloatTensor(value)

    return policy, value


def calc_loss(
    p: torch.Tensor,
    v: torch.Tensor,
    target_p: torch.Tensor,
    target_v: torch.Tensor
) -> torch.Tensor:
    policy_loss = - torch.sum(target_p * p)
    value_loss = torch.sum((target_v - v.view(-1)) ** 2)
    total_loss = policy_loss + value_loss

    return total_loss


def train(
    dnet: DynmNet,
    pnet: PredNet,
    rnet: ReprNet,
    replay_buffer: list,
    num_epochs: int = 2
):
    optimizer = optim.Adam([
        *rnet.parameters(),
        *pnet.parameters(),
        *dnet.parameters()
    ], lr=0.001)

    # to_plot = []

    for _ in trange(num_epochs):
        bacthes = [
            replay_buffer[k: k+20] for k in range(0, len(replay_buffer), 20)
        ]

        for batch in bacthes:
            loss = 0

            # Starting state
            inp = get_rnet_batch_input(batch)
            hidden_states = rnet.forward(inp)
            p, v = pnet.forward(hidden_states)

            target_p, target_v = get_target_policy_value(batch, 0)
            loss += calc_loss(p, v, target_p, target_v)

            # Unrolled state in the sampled trajectory
            for step in range(1, 5):
                s_a = get_dnet_batch_input(hidden_states, batch, step)
                hidden_states = dnet.forward(s_a)
                p, v = pnet.forward(hidden_states)

                target_p, target_v = get_target_policy_value(batch, step)
                loss += calc_loss(p, v, target_p, target_v)

            loss /= (len(batch) * 5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #     to_plot.append(loss.detach().numpy())

    # plt.plot(to_plot, label='train_loss')
    # plt.show()

    return dnet, pnet, rnet


if __name__ == "__main__":
    pnet = PredNet()
    dnet = DynmNet()
    rnet = ReprNet()
    rb = generate_replay_buffer(pnet, dnet, rnet, 100)
    rb = make_targets(rb)
    train(dnet, pnet, rnet, rb, 128)
