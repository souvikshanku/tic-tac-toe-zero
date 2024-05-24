import numpy as np

from game import evaluate, make_move, draw_board
from mcts import Node, backprop, init_root, search
from models import PredNet, ReprNet, DynmNet
from utils import rnet_input


def _get_mcts_policy(root: Node) -> np.ndarray:
    policy = np.array([c.visit_count for c in root.children])
    assert sum(policy) != 0
    return policy / sum(policy)


def _assign_rewards(examples: list, winner: int, reward: float) -> list:
    for ex in examples:
        if ex[1] == winner:
            ex[4] = reward
        else:
            ex[4] = - reward

    # let's also `sample` and `unroll` here
    n = len(examples)
    unroll_from = np.random.randint(0, n - 1)

    return examples[unroll_from:]


def generate_replay_buffer(
    pnet: PredNet,
    dnet: DynmNet,
    rnet: ReprNet,
    num_episodes: int = 10
) -> list:
    """Return replay buffer consisting of lists of the form:
        `[state, player, improved_policy, move, reward]`.
    """
    replay_buffer = []
    num_sims = 10

    for _ in range(num_episodes):
        examples = []
        state = np.zeros(9)
        player = 1

        while evaluate(state) is None:
            inp = rnet_input(state, player)
            hs = rnet.predict(inp)
            node = Node(hs, state, is_root=True, to_play=True)
            node = init_root(node, dnet, pnet)

            for _ in range(num_sims):
                path = [node]
                reward = search(node, player, path, dnet, pnet)
                backprop(reward, path)

            improved_policy = _get_mcts_policy(node)
            action = np.random.choice(
                range(len(improved_policy)),
                p=improved_policy
            )

            examples.append([
                state,
                player,
                improved_policy,
                action,
                None
            ])

            state = make_move(state.copy(), player, action)
            player = player * -1

        examples = _assign_rewards(examples, player * -1, evaluate(state))
        replay_buffer += [examples]

    return replay_buffer


if __name__ == "__main__":
    pnet = PredNet()
    dnet = DynmNet()
    rnet = ReprNet()

    rb = generate_replay_buffer(
        pnet, dnet, rnet, 2
    )

    for b in rb:
        for c in b:
            draw_board(c[0])
            print("player:", c[1])
            print("policy:", c[2])
            print("reward:", c[-1])
        print("--------------------------------------------")
