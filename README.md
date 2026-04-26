## Gomoku

This project is a Go implementation of Gomoku on a `15x15` board. It now includes:

- A desktop GUI built with Fyne
- An alpha-beta search player
- A trainable reinforcement learning player

### Rules

Two players take turns placing stones on the board. The first player to make five in a row horizontally, vertically, or diagonally wins.

### AI Players

#### Alpha-Beta

The search player uses depth-limited minimax with alpha-beta pruning and a handcrafted evaluation function.

#### Reinforcement Learning

The RL player uses a small neural move scorer trained in Go over board features such as:

- Immediate wins
- Threat blocking
- Longest line created by a move
- Center control
- Opponent winning threats after the move
- Local stone density

The latest RL model is stored at `models/rl_player.json`, and the strongest benchmarked snapshot is stored at `models/rl_player.best.json`.

### Run The GUI

```bash
go run .
```

You can choose either `Alpha-Beta` or `Reinforcement Learning` from the opponent selector in the desktop app.

### Train The RL Player

Train a fresh or existing model with:

```bash
go run . -mode train -episodes 2000
```

Useful flags:

```bash
go run . -mode train -episodes 4000 -checkpoint-every 500 -eval-games 200 -model models/rl_player.json -opponent-depth 2
```

Training reuses an existing saved model if one is already present, so running the command again continues improving the same weights.

At each checkpoint the trainer now:

- prints a progress summary
- saves the latest model to `models/rl_player.json`
- saves the best checkpoint model to `models/rl_player.best.json`
- evaluates the current model with zero exploration noise
- preserves the best model across later training runs if it still evaluates better
- uses a mixed-depth curriculum when `-opponent-depth` is above `1`

### Benchmark-Guided Optimization

Optimize the pure RL player specifically against Alpha-Beta with:

```bash
go run . -mode optimize -cycles 4 -games 40
```

Each optimization cycle can combine:

- alpha-beta imitation learning
- reinforcement learning episodes
- hard-position recovery from Alpha-Beta vs RL losses
- direct Alpha-Beta vs RL sparring imitation on the positions RL actually reaches
- direct head-to-head benchmarking against alpha-beta

The optimizer keeps the strongest RL snapshot based on real Alpha-Beta vs RL benchmark results, not just internal training metrics.

### Imitation Learning

Bootstrap the RL player from alpha-beta self-play with:

```bash
go run . -mode imitation -imitation-games 200 -teacher-depth 3
```

This teaches the RL network to mimic alpha-beta move choices before further reinforcement training.

### Hard-Position Hardening

Target positions where RL actually loses to alpha-beta with:

```bash
go run . -mode harden -hardening-games 20 -teacher-depth 3
```

This mines Alpha-Beta vs RL games, collects RL decision points from losses, and retrains the RL model on the alpha-beta move for those exact positions.

### Head-To-Head Benchmark

Measure Alpha-Beta vs RL directly with:

```bash
go run . -mode benchmark -games 40
```

### Files

- `main.go` contains the desktop GUI and mode selection
- `game/` contains board rules, move generation, win detection, and the alpha-beta player
- `rl/` contains the RL model, feature extraction, and training loop

### Notes

- The RL player is intentionally lightweight so it can be trained directly in Go without bringing in a large ML stack.
- The GUI defaults to the strongest saved RL snapshot when it is available.
