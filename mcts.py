import numpy as np
import torch

from game import get_valid_moves
from models import PredNet, ReprNet, DynmNet
from utils import dnet_input, rnet_input


def mask_illegal_moves(
    state: np.ndarray,
    policy: np.ndarray,
) -> np.ndarray:
    valid_moves = get_valid_moves(state)
    for idx in range(9):
        if idx not in valid_moves:
            policy[idx] = 0.0

    return policy / sum(policy)


class Node:
    def __init__(
        self,
        hidden_state,
        state=None,
        is_root=False,
        parent=None,
        to_play=None
    ):
        self.parent = parent
        self.state = state
        self.is_root = is_root
        self.hs = hidden_state
        self.children = [None] * 9
        self.value = 0
        self.visit_count = 0
        self.policy = None
        self.visited = False

        if to_play:
            self.to_play = to_play
        else:
            self.to_play = not self.parent.to_play

    def expand(self, dnet: DynmNet):
        for m in range(9):
            inp = dnet_input(self.hs, m)
            hs = dnet.predict(inp)
            self.children[m] = Node(hs, parent=self)


def init_root(
    node: Node,
    dnet: DynmNet,
    pnet: PredNet,
    noise: bool = False
) -> Node:
    node.visited = True
    node.visit_count = 0
    node.expand(dnet)

    policy, value = pnet.predict(node.hs)
    node.value = value[0]

    policy = torch.exp(policy)[0].detach().numpy().copy()
    if noise:
        noise = np.random.dirichlet([0.3] * len(policy))
    policy = noise * 0.25 + (1 - 0.25) * policy
    node.policy = mask_illegal_moves(node.state, policy)

    return node


def search(
    node: Node,
    player: int,
    path: list,
    dnet: DynmNet,
    pnet: PredNet
) -> torch.Tensor:
    if not node.visited:
        node.visited = True
        node.visit_count = 0
        node.expand(dnet)
        policy, value = pnet.predict(node.hs)
        node.policy = torch.exp(policy)[0]
        return value[0]

    max_u = -float("inf")
    best_move = None

    for m in range(9):
        Qsa = node.children[m].value
        Psa = node.policy[m]
        Nsa = node.children[m].visit_count

        # u = Qsa + 1 * Psa * np.sqrt(node.visit_count) / (1 + Nsa)
        c1 = 1.25
        c2 = 19652
        u = Qsa + Psa * (np.sqrt(node.visit_count) / (1 + Nsa)) * (c1 + np.log((node.visit_count + c2 + 1) / c2))  # noqa

        if node.parent is None:  # i.e., the root node
            if node.policy[m] == 0.0:
                u = -float("inf")

        if u > max_u:
            max_u = u
            best_move = m

    child: Node = node.children[best_move]
    path.append(child)
    v = search(child, - player, path, dnet, pnet)

    return - v


def backprop(reward, path):
    node = path[-1]
    while node:
        mlp = (-1) ** (node.to_play + 1)
        node.value = (node.visit_count * node.value + (reward * mlp)) / (node.visit_count + 1)  # noqa
        node.visit_count += 1
        node = node.parent


if __name__ == "__main__":
    from game import draw_board

    pnet = PredNet()
    dnet = DynmNet()
    rnet = ReprNet()

    state = np.array([1, 0, 0, -1, 0, 0, 1, 0, -1])
    draw_board(state)
    player = 1

    inp = rnet_input(state, player)
    hs = rnet.predict(inp)
    node = Node(hidden_state=hs, state=state, is_root=True, to_play=True)
    node = init_root(node, dnet, pnet)

    for _ in range(30):
        path = [node]
        reward = search(node, player, path, dnet, pnet)
        backprop(reward, path)
        print(node.value, reward)

    print(node.value)
    print([node.children[i].value for i in range(9)])
    print([node.children[1].children[i].value for i in range(9)])

    print(node.visit_count)
    print([node.children[i].visit_count for i in range(9)])
    print([node.children[1].children[i].visit_count for i in range(9)])
    print(node.policy)
