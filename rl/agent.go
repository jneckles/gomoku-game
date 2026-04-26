package rl

import (
	"encoding/json"
	"fmt"
	"gomoku/game"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

const (
	legacyFeatureCount   = 9
	tacticalFeatureCount = 20
	boardFeatureSize     = 15 * 15
	boardFeaturePlanes   = 4
	featureCount         = tacticalFeatureCount + boardFeatureSize*boardFeaturePlanes
	defaultHiddenSize    = 128
	modelVersion         = 4
)

type Agent struct {
	Version    int       `json:"version"`
	InputSize  int       `json:"input_size"`
	HiddenSize int       `json:"hidden_size"`
	W1         []float64 `json:"w1"`
	B1         []float64 `json:"b1"`
	W2         []float64 `json:"w2"`
	B2         float64   `json:"b2"`

	// Legacy field kept for backward-compatible loading of older linear models.
	Weights []float64 `json:"weights,omitempty"`
}

type Stats struct {
	Episodes     int
	Wins         int
	Losses       int
	Draws        int
	FinalEpsilon float64
}

func (s Stats) TotalGames() int {
	return s.Wins + s.Losses + s.Draws
}

func (s Stats) WinRate() float64 {
	total := s.TotalGames()
	if total == 0 {
		return 0
	}
	return float64(s.Wins) / float64(total)
}

func NewAgent() *Agent {
	rng := rand.New(rand.NewSource(7))
	agent := &Agent{
		Version:    modelVersion,
		InputSize:  featureCount,
		HiddenSize: defaultHiddenSize,
		W1:         make([]float64, featureCount*defaultHiddenSize),
		B1:         make([]float64, defaultHiddenSize),
		W2:         make([]float64, defaultHiddenSize),
	}

	scale1 := math.Sqrt(2.0 / float64(featureCount))
	scale2 := math.Sqrt(2.0 / float64(defaultHiddenSize))

	for i := range agent.W1 {
		agent.W1[i] = (rng.Float64()*2 - 1) * scale1
	}
	for i := range agent.W2 {
		agent.W2[i] = (rng.Float64()*2 - 1) * scale2
	}

	return agent
}

func (a *Agent) Clone() *Agent {
	clone := &Agent{
		Version:    a.Version,
		InputSize:  a.InputSize,
		HiddenSize: a.HiddenSize,
		W1:         append([]float64(nil), a.W1...),
		B1:         append([]float64(nil), a.B1...),
		W2:         append([]float64(nil), a.W2...),
		B2:         a.B2,
	}
	if len(a.Weights) > 0 {
		clone.Weights = append([]float64(nil), a.Weights...)
	}
	return clone
}

func Load(path string) (*Agent, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var agent Agent
	if err := json.Unmarshal(data, &agent); err != nil {
		return nil, err
	}

	if len(agent.W1) > 0 || agent.Version >= modelVersion {
		if err := agent.upgradeToCurrent(); err != nil {
			return nil, err
		}
		if err := agent.validate(); err != nil {
			return nil, err
		}
		agent.Weights = nil
		return &agent, nil
	}

	if len(agent.Weights) == legacyFeatureCount {
		return fromLegacyWeights(agent.Weights), nil
	}

	return nil, fmt.Errorf("model at %s is not a supported RL format", path)
}

func (a *Agent) Save(path string) error {
	if err := a.upgradeToCurrent(); err != nil {
		return err
	}
	if err := a.validate(); err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	payload := a.Clone()
	payload.Version = modelVersion
	payload.InputSize = featureCount
	payload.HiddenSize = len(payload.W2)
	payload.Weights = nil

	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0o644)
}

func (a *Agent) BestMove(board *game.Board, player game.Player) game.Move {
	move, _, _ := a.selectMove(board, player, 0, rand.New(rand.NewSource(time.Now().UnixNano())))
	return move
}

func (a *Agent) selectMove(board *game.Board, player game.Player, epsilon float64, rng *rand.Rand) (game.Move, []float64, float64) {
	moves := board.GenerateMoves()
	if len(moves) == 0 {
		return game.Move{Row: -1, Col: -1}, nil, 0
	}

	if epsilon > 0 && rng.Float64() < epsilon {
		move := moves[rng.Intn(len(moves))]
		features := extractFeatures(board, player, move)
		return move, features, a.predict(features)
	}

	bestMove := moves[0]
	bestFeatures := extractFeatures(board, player, bestMove)
	bestScore := a.predict(bestFeatures)

	for _, move := range moves[1:] {
		features := extractFeatures(board, player, move)
		score := a.predict(features)
		if score > bestScore {
			bestMove = move
			bestFeatures = features
			bestScore = score
		}
	}

	return bestMove, bestFeatures, bestScore
}

func (a *Agent) maxQ(board *game.Board, player game.Player) float64 {
	moves := board.GenerateMoves()
	if len(moves) == 0 {
		return 0
	}

	best := math.Inf(-1)
	for _, move := range moves {
		score := a.predict(extractFeatures(board, player, move))
		if score > best {
			best = score
		}
	}
	return best
}

func (a *Agent) predict(features []float64) float64 {
	_, _, output := a.forward(features)
	return output
}

func (a *Agent) trainTowards(features []float64, target, alpha float64) {
	hidden, _, output := a.forward(features)
	deltaOut := clamp(target-output, -2, 2)
	oldW2 := append([]float64(nil), a.W2...)

	for j := 0; j < a.HiddenSize; j++ {
		a.W2[j] += alpha * deltaOut * hidden[j]
	}
	a.B2 += alpha * deltaOut

	for j := 0; j < a.HiddenSize; j++ {
		deltaHidden := (1 - hidden[j]*hidden[j]) * oldW2[j] * deltaOut
		offset := j * a.InputSize
		for i := 0; i < a.InputSize; i++ {
			a.W1[offset+i] += alpha * deltaHidden * features[i]
		}
		a.B1[j] += alpha * deltaHidden
	}
}

func (a *Agent) forward(features []float64) ([]float64, []float64, float64) {
	hiddenRaw := make([]float64, a.HiddenSize)
	hidden := make([]float64, a.HiddenSize)

	for j := 0; j < a.HiddenSize; j++ {
		sum := a.B1[j]
		offset := j * a.InputSize
		for i := 0; i < a.InputSize; i++ {
			sum += a.W1[offset+i] * features[i]
		}
		hiddenRaw[j] = sum
		hidden[j] = math.Tanh(sum)
	}

	output := a.B2
	for j := 0; j < a.HiddenSize; j++ {
		output += a.W2[j] * hidden[j]
	}

	return hidden, hiddenRaw, output
}

func (a *Agent) validate() error {
	if a.InputSize == 0 {
		a.InputSize = featureCount
	}
	if a.HiddenSize == 0 {
		a.HiddenSize = len(a.W2)
	}
	if a.InputSize != featureCount {
		return fmt.Errorf("model input size is %d, expected %d", a.InputSize, featureCount)
	}
	if a.HiddenSize <= 0 {
		return errorsf("model hidden size must be positive")
	}
	if len(a.W1) != a.InputSize*a.HiddenSize {
		return fmt.Errorf("model w1 has %d values, expected %d", len(a.W1), a.InputSize*a.HiddenSize)
	}
	if len(a.B1) != a.HiddenSize {
		return fmt.Errorf("model b1 has %d values, expected %d", len(a.B1), a.HiddenSize)
	}
	if len(a.W2) != a.HiddenSize {
		return fmt.Errorf("model w2 has %d values, expected %d", len(a.W2), a.HiddenSize)
	}
	return nil
}

func (a *Agent) upgradeToCurrent() error {
	if a.InputSize == 0 {
		a.InputSize = featureCount
	}
	if a.HiddenSize == 0 {
		a.HiddenSize = len(a.W2)
	}
	if len(a.W1) == 0 {
		return nil
	}

	targetInput := featureCount
	targetHidden := a.HiddenSize
	if targetHidden < defaultHiddenSize {
		targetHidden = defaultHiddenSize
	}

	if a.InputSize == targetInput && a.HiddenSize == targetHidden && a.Version >= modelVersion {
		return nil
	}

	rng := rand.New(rand.NewSource(17))
	upgraded := &Agent{
		Version:    modelVersion,
		InputSize:  targetInput,
		HiddenSize: targetHidden,
		W1:         make([]float64, targetInput*targetHidden),
		B1:         make([]float64, targetHidden),
		W2:         make([]float64, targetHidden),
		B2:         a.B2,
	}

	scale1 := math.Sqrt(2.0 / float64(targetInput))
	scale2 := math.Sqrt(2.0 / float64(targetHidden))
	for i := range upgraded.W1 {
		upgraded.W1[i] = (rng.Float64()*2 - 1) * scale1 * 0.25
	}
	for i := range upgraded.W2 {
		upgraded.W2[i] = (rng.Float64()*2 - 1) * scale2 * 0.25
	}

	copyHidden := minInt(a.HiddenSize, targetHidden)
	copyInput := minInt(a.InputSize, targetInput)
	for h := 0; h < copyHidden; h++ {
		upgraded.B1[h] = a.B1[h]
		upgraded.W2[h] = a.W2[h]
		oldOffset := h * a.InputSize
		newOffset := h * targetInput
		copy(upgraded.W1[newOffset:newOffset+copyInput], a.W1[oldOffset:oldOffset+copyInput])
	}

	*a = *upgraded
	a.Weights = nil
	return nil
}

func fromLegacyWeights(weights []float64) *Agent {
	agent := NewAgent()

	for i := range agent.W1 {
		agent.W1[i] = 0
	}
	for i := range agent.B1 {
		agent.B1[i] = 0
	}
	for i := range agent.W2 {
		agent.W2[i] = 0
	}
	agent.B2 = weights[0]

	for i := 1; i < len(weights) && i-1 < agent.HiddenSize; i++ {
		hidden := i - 1
		agent.W1[hidden*agent.InputSize+i] = 1
		agent.W2[hidden] = weights[i]
	}

	return agent
}

func extractFeatures(board *game.Board, player game.Player, move game.Move) []float64 {
	opponent := player.Other()
	features := make([]float64, featureCount)
	features[0] = 1

	beforeEval := normalizeEval(board.Evaluate(player))
	beforeThreats := float64(countWinningMoves(board, opponent))
	selfThreatsBefore := float64(countWinningMoves(board, player))

	blockedThreat := 0.0
	for _, threat := range board.WinningMoves(opponent) {
		if threat.Row == move.Row && threat.Col == move.Col {
			blockedThreat = 1
			break
		}
	}

	sim := board.Clone()
	if err := sim.Place(move.Row, move.Col, player); err != nil {
		return features
	}

	immediateWin := 0.0
	if sim.HasFive(move.Row, move.Col, player) {
		immediateWin = 1
	}

	afterEval := normalizeEval(sim.Evaluate(player))
	center := float64(sim.Size - 1)
	centerBias := 1.0
	if center > 0 {
		dist := math.Abs(float64(move.Row)-center/2.0) + math.Abs(float64(move.Col)-center/2.0)
		centerBias = 1.0 - dist/center
	}

	longest := float64(sim.LongestLineAt(move.Row, move.Col, player)) / 5.0
	oppThreatsAfter := float64(countWinningMoves(sim, opponent)) / 4.0
	neighborDensity1 := float64(board.OccupiedNeighbors(move.Row, move.Col, 1)) / 8.0
	mobility := float64(len(sim.GenerateMoves())) / float64(sim.Size*sim.Size)

	selfThreatsAfter := float64(countWinningMoves(sim, player)) / 4.0
	oppLongest := float64(sim.MaxLine(opponent)) / 5.0
	neighborDensity2 := float64(board.OccupiedNeighbors(move.Row, move.Col, 2)) / 24.0
	stoneAdvantage := float64(sim.Count(player)-sim.Count(opponent)) / float64(sim.Size*sim.Size)
	oppCenterPressure := centerControl(sim, opponent)
	selfCenterPressure := centerControl(sim, player)
	maxLineGain := float64(sim.MaxLine(player)-board.MaxLine(player)) / 5.0

	features[1] = afterEval - beforeEval
	features[2] = centerBias
	features[3] = immediateWin
	features[4] = blockedThreat
	features[5] = longest
	features[6] = oppThreatsAfter
	features[7] = mobility
	features[8] = neighborDensity1 + beforeThreats/6.0
	features[9] = beforeEval
	features[10] = selfThreatsAfter - selfThreatsBefore/4.0
	features[11] = -oppLongest
	features[12] = neighborDensity2
	features[13] = stoneAdvantage
	features[14] = selfCenterPressure - oppCenterPressure
	features[15] = maxLineGain
	features[16] = float64(sim.MaxLine(player)) / 5.0
	features[17] = (beforeThreats - float64(countWinningMoves(sim, opponent))) / 4.0
	features[18] = float64(board.OccupiedNeighbors(move.Row, move.Col, 3)) / 48.0
	if blockedThreat == 1 && oppThreatsAfter == 0 {
		features[19] = 1
	} else {
		features[19] = clamp(selfThreatsAfter-oppThreatsAfter, -1, 1)
	}

	offset := tacticalFeatureCount
	for r := 0; r < sim.Size; r++ {
		for c := 0; c < sim.Size; c++ {
			idx := r*sim.Size + c
			switch sim.Get(r, c) {
			case player:
				features[offset+idx] = 1
			case opponent:
				features[offset+boardFeatureSize+idx] = 1
			default:
				features[offset+2*boardFeatureSize+idx] = 1
			}

			if r == move.Row && c == move.Col {
				features[offset+3*boardFeatureSize+idx] = 1
			}
		}
	}

	return features
}

func normalizeEval(score int) float64 {
	return math.Tanh(float64(score) / 200000.0)
}

func countWinningMoves(board *game.Board, player game.Player) int {
	return len(board.WinningMoves(player))
}

func centerControl(board *game.Board, player game.Player) float64 {
	center := float64(board.Size-1) / 2
	score := 0.0
	for r := 0; r < board.Size; r++ {
		for c := 0; c < board.Size; c++ {
			if board.Get(r, c) != player {
				continue
			}
			dist := math.Abs(float64(r)-center) + math.Abs(float64(c)-center)
			score += 1.0 - dist/float64(board.Size)
		}
	}
	return score / float64(board.Size)
}

func errorsf(msg string) error {
	return fmt.Errorf(msg)
}
