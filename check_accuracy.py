import numpy as np
import torch
from tqdm import trange

from game import evaluate, make_move
from mcts import Node, backprop, init_root, search
from models import ReprNet, PredNet, DynmNet
from replay_buffer import _get_mcts_policy
from utils import rnet_input


def play_vs_random(
    as_player: int,
    dnet: DynmNet,
    pnet: PredNet,
    rnet: ReprNet,
    num_games: int = 100
):
    wins_or_draws = 0
    loses = 0
    to_play = as_player
    num_sims = 20

    for _ in trange(num_games):
        state = np.zeros(9)
        player = 1
        trajectory = [state]
        while evaluate(state) is None:
            if player == to_play:
                inp = rnet_input(trajectory, player)
                hs = rnet.predict(inp)
                node = Node(
                    hidden_state=hs,
                    state=state,
                    is_root=True,
                    to_play=True
                )
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
                player *= -1

            else:
                z = np.where(state == 0)[0]
                action = np.random.choice(z)
                state = make_move(state.copy(), player, action)
                player = player * -1

        if evaluate(state) == 1 or 0:
            wins_or_draws += 1
        else:
            loses += 1

    return wins_or_draws, loses


if __name__ == "__main__":
    # change this to the checkpoint you want to test
    checkpoint = torch.load("checkpoints/itr_8.pt")

    pnet = PredNet()
    dnet = DynmNet()
    rnet = ReprNet()

    dnet.load_state_dict(checkpoint['dnet_state_dict'])
    pnet.load_state_dict(checkpoint['pnet_state_dict'])
    rnet.load_state_dict(checkpoint['rnet_state_dict'])

    num_games = 100

    print("Playing as X...")
    as_X, _ = play_vs_random(1, dnet, pnet, rnet, num_games)

    print("Playing as O...")
    as_O, _ = play_vs_random(-1, dnet, pnet, rnet, num_games)

    frac = (as_X + as_O) / (num_games * 2)

    print(f"""
        # games (out of {num_games}) won / drawn when played as X: {as_X})
        # games (out of {num_games}) won / drawn when played as O: {as_O})
        fraction of games won / drawn by model: {frac}
    """)
