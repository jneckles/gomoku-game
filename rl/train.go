package rl

import (
	"fmt"
	"gomoku/game"
	"math"
	"math/rand"
	"time"
)

const (
	replayCapacity = 100_000
	replayBatch    = 16
)

type experience struct {
	features []float64
	target   float64
}

type replayBuffer struct {
	data []experience
	next int
}

type TrainerConfig struct {
	BoardSize       int
	Episodes        int
	LearningRate    float64
	Discount        float64
	EpsilonStart    float64
	EpsilonEnd      float64
	OpponentDepth   int
	TeacherDepth    int
	TeacherBlend    float64
	MixedCurriculum bool
	Seed            int64
}

func DefaultTrainerConfig() TrainerConfig {
	return TrainerConfig{
		BoardSize:       15,
		Episodes:        2000,
		LearningRate:    0.001,
		Discount:        0.95,
		EpsilonStart:    0.24,
		EpsilonEnd:      0.02,
		OpponentDepth:   3,
		TeacherDepth:    3,
		TeacherBlend:    0.18,
		MixedCurriculum: true,
		Seed:            time.Now().UnixNano(),
	}
}

func Train(agent *Agent, cfg TrainerConfig) Stats {
	if cfg.Seed == 0 {
		cfg.Seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(cfg.Seed))
	stats := Stats{Episodes: cfg.Episodes}
	replay := &replayBuffer{data: make([]experience, 0, replayCapacity)}

	for episode := 0; episode < cfg.Episodes; episode++ {
		epsilon := epsilonForEpisode(cfg, episode)
		result := playTrainingEpisode(agent, cfg, epsilon, rng, replay)

		switch result {
		case 1:
			stats.Wins++
		case -1:
			stats.Losses++
		default:
			stats.Draws++
		}
		stats.FinalEpsilon = epsilon
	}

	return stats
}

// rlTrajectoryStep records one RL move during a training episode.
type rlTrajectoryStep struct {
	features   []float64
	shapingRwd float64
}

func playTrainingEpisode(agent *Agent, cfg TrainerConfig, epsilon float64, rng *rand.Rand, replay *replayBuffer) int {
	board := game.FullBoard(cfg.BoardSize)
	rlPlayer := game.P1
	opponent := game.P2
	current := game.P1

	if rng.Intn(2) == 1 {
		rlPlayer = game.P2
		opponent = game.P1
	}

	var trajectory []rlTrajectoryStep

	for !board.Full() {
		if current == rlPlayer {
			prevOppThreats := countWinningMoves(board, opponent)

			move, features, _ := agent.selectMove(board, rlPlayer, epsilon, rng)
			if move.Row == -1 {
				applyMCReturns(agent, replay, trajectory, 0.0, cfg.LearningRate, cfg.Discount, rng)
				return 0
			}

			// Inline teacher imitation — signals applied immediately, not via MC.
			teacherDepth := cfg.TeacherDepth
			if teacherDepth <= 0 {
				teacherDepth = max(1, cfg.OpponentDepth)
			}
			if shouldConsultTeacher(board, rlPlayer) {
				teacherMove := board.BestMove(rlPlayer, teacherDepth)
				if teacherMove.Row != -1 && (teacherMove.Row != move.Row || teacherMove.Col != move.Col) {
					blend := clamp(cfg.TeacherBlend, 0, 1)
					teacherFeatures := extractFeatures(board, rlPlayer, teacherMove)
					penaltyTarget := imitationNegativeTarget(board, rlPlayer, move)
					applyLearning(agent, replay, teacherFeatures, 0.85, cfg.LearningRate*blend, rng)
					applyLearning(agent, replay, features, penaltyTarget, cfg.LearningRate*blend, rng)
				}
			}

			if err := board.Place(move.Row, move.Col, rlPlayer); err != nil {
				applyMCReturns(agent, replay, trajectory, -1.0, cfg.LearningRate, cfg.Discount, rng)
				return -1
			}

			// Tactical shaping (no eval-based signal to avoid leaking AB evaluator).
			postSelfThreats := countWinningMoves(board, rlPlayer)
			postOppThreats := countWinningMoves(board, opponent)
			shaping := 0.0
			if postSelfThreats > 0 {
				shaping += 0.05 * float64(postSelfThreats)
			}
			if prevOppThreats > 0 && postOppThreats == 0 {
				shaping += 0.15
			}
			shaping -= 0.05 * float64(postOppThreats)
			shaping = clamp(shaping, -0.2, 0.2)

			trajectory = append(trajectory, rlTrajectoryStep{features: features, shapingRwd: shaping})

			if board.HasFive(move.Row, move.Col, rlPlayer) {
				applyMCReturns(agent, replay, trajectory, 1.0, cfg.LearningRate, cfg.Discount, rng)
				return 1
			}
			if board.Full() {
				applyMCReturns(agent, replay, trajectory, 0.0, cfg.LearningRate, cfg.Discount, rng)
				return 0
			}

			opponentMove := chooseOpponentMove(board, opponent, opponentDepthForEpisode(cfg, rng), rng, episodeOpponentStyle(rng))
			if opponentMove.Row == -1 {
				applyMCReturns(agent, replay, trajectory, 0.0, cfg.LearningRate, cfg.Discount, rng)
				return 0
			}
			if err := board.Place(opponentMove.Row, opponentMove.Col, opponent); err != nil {
				applyMCReturns(agent, replay, trajectory, -1.0, cfg.LearningRate, cfg.Discount, rng)
				return -1
			}
			if board.HasFive(opponentMove.Row, opponentMove.Col, opponent) {
				applyMCReturns(agent, replay, trajectory, -1.0, cfg.LearningRate, cfg.Discount, rng)
				return -1
			}
			if board.Full() {
				applyMCReturns(agent, replay, trajectory, 0.0, cfg.LearningRate, cfg.Discount, rng)
				return 0
			}

			current = rlPlayer
			continue
		}

		// Opponent's initial move (only reached when rlPlayer == P2).
		move := chooseOpponentMove(board, current, opponentDepthForEpisode(cfg, rng), rng, episodeOpponentStyle(rng))
		if move.Row == -1 {
			return 0
		}
		if err := board.Place(move.Row, move.Col, current); err != nil {
			return 1
		}
		if board.HasFive(move.Row, move.Col, current) {
			if current == opponent {
				applyMCReturns(agent, replay, trajectory, -1.0, cfg.LearningRate, cfg.Discount, rng)
				return -1
			}
			return 1
		}
		current = rlPlayer
	}

	applyMCReturns(agent, replay, trajectory, 0.0, cfg.LearningRate, cfg.Discount, rng)
	return 0
}

// applyMCReturns propagates the terminal reward backward through the trajectory.
// Each step's target = shaping_reward + discount * future_return.
func applyMCReturns(agent *Agent, replay *replayBuffer, trajectory []rlTrajectoryStep, terminalReward float64, lr, discount float64, rng *rand.Rand) {
	G := terminalReward
	for i := len(trajectory) - 1; i >= 0; i-- {
		step := trajectory[i]
		G = step.shapingRwd + discount*G
		target := clamp(G, -1.5, 1.5)
		applyLearning(agent, replay, step.features, target, lr, rng)
	}
}

func Evaluate(agent *Agent, cfg TrainerConfig, episodes int) Stats {
	if episodes <= 0 {
		return Stats{}
	}

	evalCfg := cfg
	evalCfg.Episodes = episodes
	evalCfg.EpsilonStart = 0
	evalCfg.EpsilonEnd = 0
	evalCfg.Seed = 1337

	rng := rand.New(rand.NewSource(evalCfg.Seed))
	stats := Stats{Episodes: episodes}

	for episode := 0; episode < episodes; episode++ {
		result := playTrainingEpisode(agent, evalCfg, 0, rng, nil)
		switch result {
		case 1:
			stats.Wins++
		case -1:
			stats.Losses++
		default:
			stats.Draws++
		}
	}

	return stats
}

func epsilonForEpisode(cfg TrainerConfig, episode int) float64 {
	if cfg.Episodes <= 1 {
		return cfg.EpsilonEnd
	}
	progress := float64(episode) / float64(cfg.Episodes-1)
	return cfg.EpsilonStart + progress*(cfg.EpsilonEnd-cfg.EpsilonStart)
}

func applyLearning(agent *Agent, replay *replayBuffer, features []float64, target, alpha float64, rng *rand.Rand) {
	if replay == nil {
		return
	}

	agent.trainTowards(features, target, alpha)
	replay.add(features, target)

	for _, exp := range replay.sample(rng, replayBatch) {
		agent.trainTowards(exp.features, exp.target, alpha*0.5)
	}
}

func opponentDepthForEpisode(cfg TrainerConfig, rng *rand.Rand) int {
	if cfg.OpponentDepth <= 1 {
		return 1
	}
	if !cfg.MixedCurriculum {
		return cfg.OpponentDepth
	}

	roll := rng.Float64()
	switch {
	case roll < 0.20:
		return 1
	case roll < 0.50:
		return max(1, cfg.OpponentDepth-1)
	default:
		return cfg.OpponentDepth
	}
}

func chooseOpponentMove(board *game.Board, player game.Player, depth int, rng *rand.Rand, style string) game.Move {
	moves := board.GenerateMoves()
	if len(moves) == 0 {
		return game.Move{Row: -1, Col: -1}
	}

	switch style {
	case "random":
		return moves[rng.Intn(len(moves))]
	case "mirror":
		agent := NewAgent()
		agent.Weights = []float64{0.15, 1.1, 0.45, 2.2, 2.0, 0.9, -0.8, 0.25, 0.2}
		move, _, _ := agent.selectMove(board, player, 0, rng)
		return move
	default:
		return board.BestMove(player, depth)
	}
}

func episodeOpponentStyle(rng *rand.Rand) string {
	roll := rng.Float64()
	switch {
	case roll < 0.10:
		return "random"
	case roll < 0.18:
		return "mirror"
	default:
		return "search"
	}
}

func shouldConsultTeacher(board *game.Board, player game.Player) bool {
	opponent := player.Other()
	if countWinningMoves(board, player) > 0 || countWinningMoves(board, opponent) > 0 {
		return true
	}
	if board.MaxLine(player) >= 3 || board.MaxLine(opponent) >= 3 {
		return true
	}
	if board.Count(player)+board.Count(opponent) >= 10 {
		return true
	}
	return false
}

func clamp(v, low, high float64) float64 {
	return math.Max(low, math.Min(high, v))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (r *replayBuffer) add(features []float64, target float64) {
	cloned := append([]float64(nil), features...)
	exp := experience{features: cloned, target: target}

	if len(r.data) < replayCapacity {
		r.data = append(r.data, exp)
		return
	}

	r.data[r.next] = exp
	r.next = (r.next + 1) % replayCapacity
}

func (r *replayBuffer) sample(rng *rand.Rand, n int) []experience {
	if len(r.data) == 0 || n <= 0 {
		return nil
	}

	if n > len(r.data) {
		n = len(r.data)
	}

	out := make([]experience, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, r.data[rng.Intn(len(r.data))])
	}
	return out
}

func Summary(stats Stats) string {
	total := stats.TotalGames()
	if total == 0 {
		return "no training games were played"
	}

	winRate := 100 * stats.WinRate()
	return fmt.Sprintf(
		"episodes=%d wins=%d losses=%d draws=%d win-rate=%.1f%% final-epsilon=%.3f",
		stats.Episodes,
		stats.Wins,
		stats.Losses,
		stats.Draws,
		winRate,
		stats.FinalEpsilon,
	)
}
