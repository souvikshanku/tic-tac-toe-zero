import numpy as np

from game import evaluate, make_move, draw_board
from mcts import Node, backprop, init_root, search
from models import PredNet, ReprNet, DynmNet
from utils import rnet_input


def _get_mcts_policy(root: Node) -> np.ndarray:
    policy = np.array([c.visit_count for c in root.children])
    assert sum(policy) != 0
    return policy / sum(policy)


def _assign_rewards(examples: list, _eval: int) -> list:
    if _eval != 0:
        for ex in examples:
            if ex[1] == _eval:
                ex[4] = 1.0
            else:
                ex[4] = - 1.0

    # let's also `sample` and `unroll` here
    n = len(examples)
    unroll_from = np.random.randint(0, n - 1)

    return examples[unroll_from: unroll_from + 5]


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
                0  # will be updated in `_assign_rewards`
            ])

            state = make_move(state.copy(), player, action)
            player = player * -1

        examples = _assign_rewards(examples, evaluate(state))
        replay_buffer += [examples]

    return replay_buffer


def make_targets(replay_buffer: list) -> list:
    for g in replay_buffer:
        state = g[-1][0]
        player = g[-1][1]
        action = g[-1][3]
        state = make_move(state.copy(), player, action)
        while len(g) < 5:
            g.append([
                state,
                player * -1,
                np.array([1/9] * 9),  # uniform
                np.random.choice(9),
                0
            ])
    return replay_buffer


if __name__ == "__main__":
    pnet = PredNet()
    dnet = DynmNet()
    rnet = ReprNet()

    rb = generate_replay_buffer(
        pnet, dnet, rnet, 5
    )

    rb = make_targets(rb)

    for b in rb:
        for c in b:
            draw_board(c[0])
            print("player:", c[1])
            print("policy:", c[2])
            print("action:", c[3])
            print("reward:", c[-1])
        print("--------------------------------------------")
