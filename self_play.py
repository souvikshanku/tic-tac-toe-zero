import copy

# from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import trange

from game import evaluate, make_move
from mcts import Node, backprop, init_root, search
from models import ReprNet, PredNet, DynmNet
from replay_buffer import (
    generate_replay_buffer, _get_mcts_policy, make_targets
)
from train import train
from utils import rnet_input


def _pit_nns(
    nets1: list,
    nets2: list,
    num_episodes: int = 20,
) -> float:
    num_sims = 25
    num_wins = 0
    num_draws = 0
    win_as_O = 0
    win_as_X = 0

    def player_nets(player: int, episode_count: int) -> list:
        # play as black in the first half
        if episode_count < num_episodes / 2:
            if player == 1:
                return nets1
            else:
                return nets2
        # play as white in the second half
        elif player == 1:
            return nets2
        else:
            return nets1

    for i in trange(num_episodes, ascii=' >='):
        state = np.zeros(9)
        player = 1
        trajectory = [state]

        while True:
            dnet, pnet, rnet = player_nets(player, i)
            inp = rnet_input(trajectory, player)
            hs = rnet.predict(inp)
            node = Node(hs, state, is_root=True, to_play=True)
            node = init_root(node, dnet, pnet, True)

            for _ in range(num_sims):
                path = [node]
                reward = search(node, player, path, dnet, pnet)
                backprop(reward, path)

            improved_policy = _get_mcts_policy(node)

            action = np.random.choice(
                range(len(improved_policy)),
                p=improved_policy
            )

            state = make_move(state.copy(), player, action)
            trajectory.append(state)

            if evaluate(state) is not None:
                if evaluate(state) == 0:
                    num_draws += 1
                elif i < num_episodes / 2 and player == -1:
                    win_as_O += 1
                    num_wins += 1
                elif i >= num_episodes / 2 and player == 1:
                    win_as_X += 1
                    num_wins += 1
                break

            player = player * -1

    print(f"As 'O': {win_as_O} out of {num_episodes // 2}")
    print(f"As 'X': {win_as_X} out of {num_episodes // 2}")
    print("Total Wins: ", num_wins)
    print("Total Draws: ", num_draws)

    return (num_wins + num_draws) / num_episodes


def learn_by_self_play(num_iters: int):
    num_games = 300
    num_episodes = 40
    num_epochs = 200
    batch_size = 64
    threshold = 0.55

    pnet = PredNet()
    dnet = DynmNet()
    rnet = ReprNet()

    games = []

    for i in range(num_iters):
        print(f"Iteration: {i + 1}")

        dnet_t, pnet_t, rnet_t = copy.deepcopy([dnet, pnet, rnet])

        print("Going though self-play...")
        rb = generate_replay_buffer(pnet, dnet, rnet, num_games)
        games += make_targets(rb)

        print("Training networks...")
        dnet_t, pnet_t, rnet_t = train(
            dnet_t,
            pnet_t,
            rnet_t,
            games,
            num_epochs,
            batch_size
        )

        print("Playing against older self...")
        frac = _pit_nns(
            [dnet, pnet, rnet],
            [dnet_t, pnet_t, rnet_t],
            num_episodes
        )
        print("frac: ", frac,)

        torch.save({
            "iter": i,
            "dnet_state_dict": dnet_t.state_dict(),
            "pnet_state_dict": pnet_t.state_dict(),
            "rnet_state_dict": rnet_t.state_dict(),
        }, f"checkpoints/itr_{i+1}.pt")
        print("Checkpint saved...")

        _pit_nns(
            [DynmNet(), PredNet(), ReprNet()],
            [dnet_t, pnet_t, rnet_t]
        )

        print("---------------------------------------------")

        if frac >= threshold:
            dnet, pnet, rnet = dnet_t, pnet_t, rnet_t
            games = []

    return dnet, pnet, rnet


if __name__ == "__main__":
    pnet = PredNet()
    dnet = DynmNet()
    rnet = ReprNet()

    dnet_t, pnet_t, rnet_t = learn_by_self_play(num_iters=50)
