# tic-tac-toe-zero

1. rnet: (board, player) -> hidden state
2. dnet: (hidden state, action) -> next hidden state
3. pnet: (hidden state) -> policy, value

rnet:
    `ohe X + ohe O + [player, player, player]` -> 21
    `3 x 3` hidden state

dnet:
    `3 x 3 hs + ohe move` -> 18
    `3 x 3` hs

pnet:
    `3 x 3` hs
    ohe moves + value
