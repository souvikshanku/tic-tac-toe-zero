import numpy as np
import torch

from game import get_valid_moves
from models import PredNet, ReprNet, DynmNet
from utils import dnet_input, rnet_input


def mask_illegal_moves(
    state: np.ndarray,
    policy: torch.Tensor,
) -> np.ndarray:
    policy = policy.detach().numpy().copy()
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
    pnet: PredNet
) -> Node:
    node.visited = True
    node.visit_count = 0
    node.expand(dnet)
    policy, value = pnet.predict(node.hs)
    node.value = value
    node.policy = mask_illegal_moves(node.state, torch.exp(policy))
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
        node.policy = torch.exp(policy)
        return value

    max_u = -float("inf")
    best_move = None

    for m in range(9):
        Qsa = node.children[m].value
        Psa = node.policy[m]
        Nsa = node.children[m].visit_count

        u = Qsa + 1 * Psa * np.sqrt(node.visit_count) / (1 + Nsa)

        if node.parent is None:  # i.e., the root node
            if node.policy[m] == 0.0:
                u = -float("inf")

        if u > max_u:
            max_u = u
            best_move = m

    # if best_move is None:
    #     from game import draw_board
    #     draw_board(node.state)
    #     print(player)

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

    state = np.array([0, 0, 0, 0, 0, 0, ])
    draw_board(state)
    player = -1

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
    print([node.children[5].children[i].value for i in range(9)])

    print(node.visit_count)
    print([node.children[i].visit_count for i in range(9)])
    print([node.children[5].children[i].visit_count for i in range(9)])
