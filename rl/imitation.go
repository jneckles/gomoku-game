package rl

import (
	"fmt"
	"gomoku/game"
	"math"
	"math/rand"
	"sort"
	"time"
)

type ImitationConfig struct {
	BoardSize       int
	Games           int
	TeacherDepth    int
	NegativeSamples int
	LearningRate    float64
	Seed            int64
}

type ImitationStats struct {
	GamesTrained      int
	PositionsTrained  int
	TeacherMovesTaken int
}

type imitationPosition struct {
	board  *game.Board
	player game.Player
}

func DefaultImitationConfig() ImitationConfig {
	return ImitationConfig{
		BoardSize:       15,
		Games:           200,
		TeacherDepth:    3,
		NegativeSamples: 4,
		LearningRate:    0.03,
		Seed:            time.Now().UnixNano(),
	}
}

func TrainByImitation(agent *Agent, cfg ImitationConfig) ImitationStats {
	if cfg.Seed == 0 {
		cfg.Seed = time.Now().UnixNano()
	}
	if cfg.Games <= 0 {
		return ImitationStats{}
	}
	if cfg.NegativeSamples <= 0 {
		cfg.NegativeSamples = 4
	}

	rng := rand.New(rand.NewSource(cfg.Seed))
	stats := ImitationStats{GamesTrained: cfg.Games}

	for gameIndex := 0; gameIndex < cfg.Games; gameIndex++ {
		board := game.FullBoard(cfg.BoardSize)
		current := game.P1
		openingPlies := rng.Intn(4)

		for ply := 0; !board.Full(); ply++ {
			moves := board.GenerateMoves()
			if len(moves) == 0 {
				break
			}

			var teacherMove game.Move
			if ply < openingPlies {
				teacherMove = chooseOpeningTeacherMove(board, current, rng)
			} else {
				teacherMove = board.BestMove(current, cfg.TeacherDepth)
			}

			if teacherMove.Row == -1 {
				break
			}

			trainOnTeacherMove(agent, board, current, teacherMove, cfg.NegativeSamples, cfg.LearningRate, rng)
			stats.PositionsTrained++
			stats.TeacherMovesTaken++

			if err := board.Place(teacherMove.Row, teacherMove.Col, current); err != nil {
				break
			}
			if board.HasFive(teacherMove.Row, teacherMove.Col, current) {
				break
			}

			current = current.Other()
		}
	}

	return stats
}

func chooseOpeningTeacherMove(board *game.Board, player game.Player, rng *rand.Rand) game.Move {
	moves := board.GenerateMoves()
	if len(moves) == 0 {
		return game.Move{Row: -1, Col: -1}
	}
	if len(moves) == 1 {
		return moves[0]
	}

	type scoredMove struct {
		move  game.Move
		score float64
	}

	scored := make([]scoredMove, 0, len(moves))
	for _, move := range moves {
		sim := board.Clone()
		if err := sim.Place(move.Row, move.Col, player); err != nil {
			continue
		}

		score := float64(sim.Evaluate(player))
		score += float64(sim.LongestLineAt(move.Row, move.Col, player)) * 2500
		score -= float64(len(sim.WinningMoves(player.Other()))) * 4000
		center := float64(board.Size-1) / 2
		dist := math.Abs(float64(move.Row)-center) + math.Abs(float64(move.Col)-center)
		score -= dist * 50
		score += rng.Float64() * 40
		scored = append(scored, scoredMove{move: move, score: score})
	}

	if len(scored) == 0 {
		return moves[rng.Intn(len(moves))]
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	top := minInt(3, len(scored))
	return scored[rng.Intn(top)].move
}

func sampleNegativeMoves(moves []game.Move, best game.Move, count int, rng *rand.Rand) []game.Move {
	candidates := make([]game.Move, 0, len(moves))
	for _, move := range moves {
		if move.Row == best.Row && move.Col == best.Col {
			continue
		}
		candidates = append(candidates, move)
	}

	if len(candidates) <= count {
		return candidates
	}

	out := make([]game.Move, 0, count)
	perm := rng.Perm(len(candidates))
	for _, idx := range perm[:count] {
		out = append(out, candidates[idx])
	}
	return out
}

func imitationNegativeTarget(board *game.Board, player game.Player, move game.Move) float64 {
	sim := board.Clone()
	if err := sim.Place(move.Row, move.Col, player); err != nil {
		return -1
	}

	if sim.HasFive(move.Row, move.Col, player) {
		return 0.75
	}
	if len(sim.WinningMoves(player.Other())) > 0 {
		return -0.9
	}

	score := normalizeEval(sim.Evaluate(player))
	return clamp(score-0.2, -0.8, 0.4)
}

func ImitationSummary(stats ImitationStats) string {
	return fmt.Sprintf(
		"games=%d positions=%d teacher-moves=%d",
		stats.GamesTrained,
		stats.PositionsTrained,
		stats.TeacherMovesTaken,
	)
}

func trainOnTeacherMove(agent *Agent, board *game.Board, player game.Player, teacherMove game.Move, negativeSamples int, learningRate float64, rng *rand.Rand) {
	for _, symmetry := range chooseSymmetries(rng.Intn) {
		transformedBoard, transformedMove := transformBoardAndMove(board, teacherMove, symmetry)
		moves := transformedBoard.GenerateMoves()
		teacherFeatures := extractFeatures(transformedBoard, player, transformedMove)
		agent.trainTowards(teacherFeatures, 1.0, learningRate)

		negativeMoves := sampleNegativeMoves(moves, transformedMove, negativeSamples, rng)
		for _, neg := range negativeMoves {
			target := imitationNegativeTarget(transformedBoard, player, neg)
			agent.trainTowards(extractFeatures(transformedBoard, player, neg), target, learningRate*0.7)
		}
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
