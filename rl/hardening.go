package rl

import (
	"fmt"
	"gomoku/game"
	"math/rand"
	"time"
)

type HardeningConfig struct {
	BoardSize       int
	Games           int
	TeacherDepth    int
	NegativeSamples int
	LearningRate    float64
	Seed            int64
}

type HardeningStats struct {
	GamesProcessed     int
	RLLossesObserved   int
	PositionsRecovered int
}

func DefaultHardeningConfig() HardeningConfig {
	return HardeningConfig{
		BoardSize:       15,
		Games:           20,
		TeacherDepth:    3,
		NegativeSamples: 4,
		LearningRate:    0.001,
		Seed:            time.Now().UnixNano(),
	}
}

func TrainOnHardPositions(agent *Agent, cfg HardeningConfig) HardeningStats {
	if cfg.Seed == 0 {
		cfg.Seed = time.Now().UnixNano()
	}
	if cfg.Games <= 0 {
		return HardeningStats{}
	}
	if cfg.NegativeSamples <= 0 {
		cfg.NegativeSamples = 4
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	stats := HardeningStats{GamesProcessed: cfg.Games}

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
		rlPositions := make([]imitationPosition, 0, 64)

		for ply := 0; !board.Full(); ply++ {
			if current == rlPlayer {
				rlPositions = append(rlPositions, imitationPosition{
					board:  board.Clone(),
					player: current,
				})
			}

			var move game.Move
			if ply < openingPlies {
				move = chooseOpeningTeacherMove(board, current, rng)
			} else if current == alphaPlayer {
				move = board.BestMove(current, cfg.TeacherDepth)
			} else {
				move = agent.BestMove(board, current)
			}

			if move.Row == -1 || move.Col == -1 {
				break
			}
			if err := board.Place(move.Row, move.Col, current); err != nil {
				break
			}

			if board.HasFive(move.Row, move.Col, current) {
				if current == alphaPlayer {
					stats.RLLossesObserved++
					for _, pos := range rlPositions {
						teacherMove := pos.board.BestMove(pos.player, cfg.TeacherDepth)
						if teacherMove.Row == -1 {
							continue
						}
						trainOnTeacherMove(agent, pos.board, pos.player, teacherMove, cfg.NegativeSamples, cfg.LearningRate, rng)
						stats.PositionsRecovered++
					}
				}
				break
			}

			current = current.Other()
		}
	}

	return stats
}

func HardeningSummary(stats HardeningStats) string {
	return fmt.Sprintf(
		"games=%d rl-losses=%d recovered-positions=%d",
		stats.GamesProcessed,
		stats.RLLossesObserved,
		stats.PositionsRecovered,
	)
}
