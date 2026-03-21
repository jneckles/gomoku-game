package game

// these values are used for the win and loss scoring for alpha beta
const (
	PosInf = 1_000_000_000
	NegInf = -1_000_000_000
)

// this function returns the best move for the agent using alpha beta pruning
// the depth controls how far ahead the AI searches.
// higher depth makes it stronger but slower
func (b *Board) BestMove(aiPlayer Player, depth int) Move {
	moves := b.GenerateMoves()

	// just a safety incase no moves exist.
	if len(moves) == 0 {
		return Move{Row: -1, Col: -1}
	}

	bestMove := moves[0]
	bestScore := NegInf

	for _, move := range moves {
		// this tries the AI move
		if err := b.Place(move.Row, move.Col, aiPlayer); err != nil {
			continue
		}

		var score int

		// if the move wins immediately then it's taken
		if b.HasFive(move.Row, move.Col, aiPlayer) {
			score = PosInf
		} else {
			score = b.alphaBeta(depth-1, NegInf, PosInf, false, aiPlayer, move)
		}

		// undos the possible move that's taken
		b.Remove(move.Row, move.Col)

		if score > bestScore {
			bestScore = score
			bestMove = move
		}
	}
	return bestMove
}

// this alpha beta function does optimized mini max with the alpha beta pruning step
// when maximizing is true then it's the AI's turn
// when minimizing is true then it's the human's turn.
// lastMove is the move that was made just before getting to the current position
// this helps to check if that move ended the game.
func (b *Board) alphaBeta(depth int, alpha int, beta int, maximizing bool, aiPlayer Player, lastMove Move) int {
	opponent := aiPlayer.Other()

	// checks the terminal state from the last move that was played.
	lastPlayer := opponent
	if !maximizing {
		// if it's the minimizing turn now then the Ai made the last move.
		lastPlayer = aiPlayer
	}

	if lastMove.Row != -1 && lastMove.Col != -1 {
		if b.HasFive(lastMove.Row, lastMove.Col, lastPlayer) {
			if lastPlayer == aiPlayer {
				return PosInf
			}
			return NegInf
		}
	}

	// this signifies a draw or a limit with the depth.
	if depth == 0 || b.Full() {
		return b.Evaluate(aiPlayer)
	}

	moves := b.GenerateMoves()

	if len(moves) == 0 {
		return b.Evaluate(aiPlayer)
	}

	if maximizing {
		value := NegInf

		for _, move := range moves {
			if err := b.Place(move.Row, move.Col, aiPlayer); err != nil {
				continue
			}

			score := b.alphaBeta(depth-1, alpha, beta, false, aiPlayer, move)

			b.Remove(move.Row, move.Col)

			if score > value {
				value = score
			}
			if value > alpha {
				alpha = value
			}
			// pruning step
			if alpha >= beta {
				break
			}
		}

		return value
	}
	value := PosInf

	for _, move := range moves {
		if err := b.Place(move.Row, move.Col, opponent); err != nil {
			continue
		}

		score := b.alphaBeta(depth-1, alpha, beta, true, aiPlayer, move)

		b.Remove(move.Row, move.Col)

		if score < value {
			value = score
		}
		if value < beta {
			beta = value
		}
		// pruning step
		if beta <= alpha {
			break
		}
	}
	return value
}
