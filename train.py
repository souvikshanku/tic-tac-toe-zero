# from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from models import ReprNet, PredNet, DynmNet
from replay_buffer import generate_replay_buffer, make_targets
from utils import dnet_input


def get_rnet_batch_input(rb: list) -> torch.Tensor:
    batch_inp_r = []

    for game in rb:
        inp = game[0][0]
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
    num_epochs: int = 2,
    batch_size: int = 64,
    num_unroll_steps: int = 9
):
    # to_plot = []

    optimizer = optim.Adam([
        *rnet.parameters(),
        *pnet.parameters(),
        *dnet.parameters()
    ], lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    rb_size = 300
    if len(replay_buffer) > rb_size:
        np.random.shuffle(replay_buffer)
        replay_buffer = replay_buffer[:rb_size]

    for epoch in trange(num_epochs, ascii=' >='):
        bacthes = [
            replay_buffer[k: k+batch_size]
            for k in range(0, len(replay_buffer), batch_size)
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
            for step in range(1, num_unroll_steps):
                s_a = get_dnet_batch_input(hidden_states, batch, step)
                hidden_states = dnet.forward(s_a)
                p, v = pnet.forward(hidden_states)

                target_p, target_v = get_target_policy_value(batch, step)
                loss += calc_loss(p, v, target_p, target_v)

            loss /= (len(batch) * num_unroll_steps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 500 == 0:
            scheduler.step()

    #     to_plot.append(loss.detach().numpy())

    # plt.plot(to_plot, label='train_loss')
    # plt.show()

    return dnet, pnet, rnet


if __name__ == "__main__":
    pnet = PredNet()
    dnet = DynmNet()
    rnet = ReprNet()
    rb = generate_replay_buffer(pnet, dnet, rnet, 200)
    rb = make_targets(rb)
    train(dnet, pnet, rnet, rb, 300)
