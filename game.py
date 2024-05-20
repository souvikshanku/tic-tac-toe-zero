import numpy as np


BOARD = list("""
    +-----+-----+-----+
 3  |     |     |     |
    +-----+-----+-----+
 2  |     |     |     |
    +-----+-----+-----+
 1  |     |     |     |
    +-----+-----+-----+
       a     b     c
""")


def draw_board(state: np.ndarray) -> None:
    board = BOARD.copy()

    pos = [32, 38, 44, 80, 86, 92, 128, 134, 140]

    for i, move in enumerate(state):
        if move == 1:
            board[pos[i]] = "X"
        elif move == -1:
            board[pos[i]] = "O"

    print("".join(board))


def evaluate(state: np.ndarray):
    s = state.reshape(3, 3)

    row_s = s.sum(axis=0)
    col_s = s.sum(axis=1)
    if 3 in row_s or 3 in col_s:
        return 1
    if -3 in row_s or -3 in col_s:
        return -1

    diag_s1 = np.diag(s).sum()
    diag_s2 = np.flipud(s).diagonal().sum()
    if 3 == [diag_s1, diag_s2]:
        return 1
    elif -3 in [diag_s1, diag_s2]:
        return -1

    # Draw
    if 0 not in state:
        return 0

    else:
        return None


def make_move(
    state: np.ndarray,
    player: int,
    at: int
) -> np.ndarray:
    assert state[at] == 0
    state[at] = player
    return state


def get_valid_moves(state: np.ndarray) -> np.ndarray | None:
    if evaluate(state) is None:
        return np.where(state == 0)[0]
    else:
        print("Game Over.")


if __name__ == "__main__":
    state = np.array([-1, 0, -1, 0, -1, 0, 1, 1, 0])
    draw_board(state)
    print(get_valid_moves(state))
    draw_board(make_move(state, -1, 8))
    print(evaluate(state))
