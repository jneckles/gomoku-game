package rl

import (
	"fmt"
	"gomoku/game"
	"math/rand"
	"time"
)

type SparringConfig struct {
	BoardSize       int
	Games           int
	TeacherDepth    int
	NegativeSamples int
	LearningRate    float64
	Seed            int64
}

type SparringStats struct {
	GamesPlayed      int
	PositionsTrained int
	RLWins           int
	RLLosses         int
	Draws            int
}

func DefaultSparringConfig() SparringConfig {
	return SparringConfig{
		BoardSize:       15,
		Games:           12,
		TeacherDepth:    3,
		NegativeSamples: 4,
		LearningRate:    0.025,
		Seed:            time.Now().UnixNano(),
	}
}

func TrainBySparring(agent *Agent, cfg SparringConfig) SparringStats {
	if cfg.Seed == 0 {
		cfg.Seed = time.Now().UnixNano()
	}
	if cfg.Games <= 0 {
		return SparringStats{}
	}
	if cfg.NegativeSamples <= 0 {
		cfg.NegativeSamples = 4
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	stats := SparringStats{GamesPlayed: cfg.Games}

	for gameIndex := 0; gameIndex < cfg.Games; gameIndex++ {
		board := game.FullBoard(cfg.BoardSize)
		alphaPlayer := game.P1
		rlPlayer := game.P2
		if gameIndex%2 == 1 {
			alphaPlayer = game.P2
			rlPlayer = game.P1
		}

		current := game.P1
		openingPlies := rng.Intn(4)

		for ply := 0; !board.Full(); ply++ {
			var move game.Move
			if ply < openingPlies {
				move = chooseOpeningTeacherMove(board, current, rng)
			} else if current == alphaPlayer {
				move = board.BestMove(current, cfg.TeacherDepth)
			} else {
				teacherMove := board.BestMove(current, cfg.TeacherDepth)
				if teacherMove.Row == -1 {
					stats.Draws++
					break
				}

				rlMove := agent.BestMove(board, current)
				if rlMove.Row != -1 && (rlMove.Row != teacherMove.Row || rlMove.Col != teacherMove.Col) {
					target := imitationNegativeTarget(board, current, rlMove)
					agent.trainTowards(extractFeatures(board, current, rlMove), target, cfg.LearningRate*0.9)
				}
				trainOnTeacherMove(agent, board, current, teacherMove, cfg.NegativeSamples, cfg.LearningRate, rng)
				stats.PositionsTrained++
				move = agent.BestMove(board, current)
			}

			if move.Row == -1 || move.Col == -1 {
				stats.Draws++
				break
			}
			if err := board.Place(move.Row, move.Col, current); err != nil {
				stats.Draws++
				break
			}

			if board.HasFive(move.Row, move.Col, current) {
				switch current {
				case rlPlayer:
					stats.RLWins++
				case alphaPlayer:
					stats.RLLosses++
				}
				break
			}
			if board.Full() {
				stats.Draws++
				break
			}

			current = current.Other()
		}
	}

	return stats
}

func SparringSummary(stats SparringStats) string {
	total := stats.GamesPlayed
	if total == 0 {
		return "games=0"
	}

	return fmt.Sprintf(
		"games=%d positions=%d rl-wins=%d rl-losses=%d draws=%d rl-win-rate=%.1f%%",
		stats.GamesPlayed,
		stats.PositionsTrained,
		stats.RLWins,
		stats.RLLosses,
		stats.Draws,
		100*float64(stats.RLWins)/float64(total),
	)
}
