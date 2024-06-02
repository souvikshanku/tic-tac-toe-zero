# tic-tac-toe-zero

Implementation of [MuZero](https://arxiv.org/pdf/1911.08265) for Tic-Tac-Toe.

It can play _optimally_ 65-70% of the times if you train long enough **and** if you are lucky.

[RL is hard](https://www.alexirpan.com/2018/02/14/rl-hard.html) (ToT)

## Example Usage

```bash
git clone https://github.com/souvikshanku/tic-tac-toe-zero.git
cd tic-tac-toe-zero

python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

# End-to-end training
python3 self_play.py

# Play against 'random' agent
python3 check_accuracy.py
```
