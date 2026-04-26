package rl

import (
	"fmt"
	"gomoku/game"
	"math"
	"math/rand"
	"time"
)

const (
	replayCapacity = 16384
	replayBatch    = 24
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
		LearningRate:    0.05,
		Discount:        0.94,
		EpsilonStart:    0.24,
		EpsilonEnd:      0.02,
		OpponentDepth:   1,
		TeacherDepth:    2,
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

func playTrainingEpisode(agent *Agent, cfg TrainerConfig, epsilon float64, rng *rand.Rand, replay *replayBuffer) int {
	board := game.FullBoard(cfg.BoardSize)
	rlPlayer := game.P1
	opponent := game.P2
	current := game.P1

	if rng.Intn(2) == 1 {
		rlPlayer = game.P2
		opponent = game.P1
	}

	for !board.Full() {
		if current == rlPlayer {
			prevEval := normalizeEval(board.Evaluate(rlPlayer))
			prevSelfThreats := countWinningMoves(board, rlPlayer)
			prevOppThreats := countWinningMoves(board, opponent)
			prevSelfLine := board.MaxLine(rlPlayer)
			prevOppLine := board.MaxLine(opponent)
			move, features, _ := agent.selectMove(board, rlPlayer, epsilon, rng)
			if move.Row == -1 {
				return 0
			}

			teacherDepth := cfg.TeacherDepth
			if teacherDepth <= 0 {
				teacherDepth = max(1, cfg.OpponentDepth)
			}
			if shouldConsultTeacher(board, rlPlayer) {
				teacherMove := board.BestMove(rlPlayer, teacherDepth)
				if teacherMove.Row != -1 && (teacherMove.Row != move.Row || teacherMove.Col != move.Col) {
					teacherFeatures := extractFeatures(board, rlPlayer, teacherMove)
					teacherTarget := 0.85
					penaltyTarget := imitationNegativeTarget(board, rlPlayer, move)
					blend := clamp(cfg.TeacherBlend, 0, 1)
					applyLearning(agent, replay, teacherFeatures, teacherTarget, cfg.LearningRate*blend, rng)
					applyLearning(agent, replay, features, penaltyTarget, cfg.LearningRate*blend, rng)
				}
			}

			if err := board.Place(move.Row, move.Col, rlPlayer); err != nil {
				return -1
			}

			if board.HasFive(move.Row, move.Col, rlPlayer) {
				target := 1.0
				applyLearning(agent, replay, features, target, cfg.LearningRate, rng)
				return 1
			}
			if board.Full() {
				applyLearning(agent, replay, features, 0, cfg.LearningRate, rng)
				return 0
			}

			opponentMove := chooseOpponentMove(board, opponent, opponentDepthForEpisode(cfg, rng), rng, episodeOpponentStyle(rng))
			if opponentMove.Row == -1 {
				applyLearning(agent, replay, features, 0, cfg.LearningRate, rng)
				return 0
			}
			if err := board.Place(opponentMove.Row, opponentMove.Col, opponent); err != nil {
				applyLearning(agent, replay, features, -1, cfg.LearningRate, rng)
				return -1
			}
			if board.HasFive(opponentMove.Row, opponentMove.Col, opponent) {
				applyLearning(agent, replay, features, -1, cfg.LearningRate, rng)
				return -1
			}
			if board.Full() {
				applyLearning(agent, replay, features, 0, cfg.LearningRate, rng)
				return 0
			}

			nextQ := agent.maxQ(board, rlPlayer)
			postEval := normalizeEval(board.Evaluate(rlPlayer))
			postSelfThreats := countWinningMoves(board, rlPlayer)
			postOppThreats := countWinningMoves(board, opponent)
			postSelfLine := board.MaxLine(rlPlayer)
			postOppLine := board.MaxLine(opponent)

			reward := clamp(postEval-prevEval, -0.2, 0.2)
			reward += clamp(0.12*float64(postSelfThreats-prevSelfThreats), -0.18, 0.18)
			reward -= clamp(0.16*float64(postOppThreats-prevOppThreats), -0.22, 0.22)
			reward += clamp(0.05*float64(postSelfLine-prevSelfLine), -0.10, 0.10)
			reward -= clamp(0.06*float64(postOppLine-prevOppLine), -0.10, 0.10)
			if prevOppThreats > 0 && postOppThreats == 0 {
				reward += 0.15
			}
			if postSelfThreats > prevSelfThreats {
				reward += 0.08
			}
			reward = clamp(reward, -0.45, 0.45)
			target := reward + cfg.Discount*nextQ
			applyLearning(agent, replay, features, target, cfg.LearningRate, rng)
			current = rlPlayer
			continue
		}

		move := chooseOpponentMove(board, current, opponentDepthForEpisode(cfg, rng), rng, episodeOpponentStyle(rng))
		if move.Row == -1 {
			return 0
		}
		if err := board.Place(move.Row, move.Col, current); err != nil {
			return 1
		}
		if board.HasFive(move.Row, move.Col, current) {
			if current == opponent {
				return -1
			}
			return 1
		}
		current = rlPlayer
	}

	return 0
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
		agent.trainTowards(exp.features, exp.target, alpha*0.6)
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
	case roll < 0.30:
		return 1
	case roll < 0.65:
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
