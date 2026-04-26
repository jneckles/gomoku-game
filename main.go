package main

import (
	"errors"
	"flag"
	"fmt"
	"gomoku/game"
	"gomoku/rl"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"
)

const (
	boardSize        = 15
	searchDepth      = 3
	rlSearchDepth    = 4 // searched RL now looks at least as deep as the alpha-beta baseline
	defaultModelPath = "models/rl_player.json"
	aiAlphaBeta      = "Alpha-Beta"
	aiRL             = "Reinforcement Learning"
	modeHumanVsAB    = "Human vs Alpha-Beta"
	modeHumanVsRL    = "Human vs Reinforcement Learning"
	modeABVsRL       = "Alpha-Beta vs Reinforcement Learning"
	modeGUI          = "gui"
	modeTrain        = "train"
	modeBenchmark    = "benchmark"
	modeImitation    = "imitation"
	modeHarden       = "harden"
	modeOptimize     = "optimize"
)

type benchmarkStats struct {
	Games          int
	AlphaBetaWins  int
	RLWins         int
	Draws          int
	AlphaBetaFirst int
	RLFirst        int
}

type gomokuApp struct {
	board       *game.Board
	humanPlayer game.Player
	aiPlayer    game.Player
	current     game.Player
	status      string
	gameOver    bool
	lastMove    *game.Move

	modelPath    string
	selectedAI   string
	matchMode    string
	rlAgent      *rl.Agent
	rlModelError error

	app          fyne.App
	window       fyne.Window
	statusLabel  *widget.Label
	infoLabel    *widget.Label
	buttons      [][]*widget.Button
	lastMoveText *widget.Label
	aiSelect     *widget.Select
	matchSelect  *widget.Select
	stepButton   *widget.Button
	autoButton   *widget.Button
	syncingUI    bool
	rng          *rand.Rand
	aiMatchPlies int
	openingPlies int
}

func main() {
	mode := flag.String("mode", modeGUI, "mode to run: gui, train, benchmark, imitation, harden, or optimize")
	aiMode := flag.String("ai", aiAlphaBeta, "AI opponent for GUI: 'Alpha-Beta' or 'Reinforcement Learning'")
	modelPath := flag.String("model", defaultModelPath, "path to the RL model file")
	episodes := flag.Int("episodes", rl.DefaultTrainerConfig().Episodes, "training episodes for RL mode")
	opponentDepth := flag.Int("opponent-depth", rl.DefaultTrainerConfig().OpponentDepth, "search depth used by the training opponent")
	checkpointEvery := flag.Int("checkpoint-every", 500, "save/report training progress every N episodes")
	evalGames := flag.Int("eval-games", 200, "number of zero-exploration evaluation games to run per checkpoint")
	benchmarkGames := flag.Int("games", 40, "number of head-to-head benchmark games")
	imitationGames := flag.Int("imitation-games", rl.DefaultImitationConfig().Games, "number of alpha-beta self-play games for imitation learning")
	teacherDepth := flag.Int("teacher-depth", searchDepth, "alpha-beta depth used as the imitation teacher")
	hardeningGames := flag.Int("hardening-games", rl.DefaultHardeningConfig().Games, "number of Alpha-Beta vs RL games to mine for hard-position imitation")
	sparringGames := flag.Int("sparring-games", rl.DefaultSparringConfig().Games, "number of direct Alpha-Beta vs RL sparring games to imitate during optimization")
	optimizeCycles := flag.Int("cycles", 4, "number of benchmark-guided pure-RL optimization cycles")
	flag.Parse()

	switch *mode {
	case modeTrain:
		if err := runTraining(*modelPath, *episodes, *opponentDepth, *checkpointEvery, *evalGames); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case modeBenchmark:
		if err := runBenchmark(*modelPath, *benchmarkGames); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case modeImitation:
		if err := runImitation(*modelPath, *imitationGames, *teacherDepth); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case modeHarden:
		if err := runHardening(*modelPath, *hardeningGames, *teacherDepth); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case modeOptimize:
		if err := runOptimize(*modelPath, *optimizeCycles, *episodes, *opponentDepth, *checkpointEvery, *evalGames, *imitationGames, *hardeningGames, *sparringGames, *teacherDepth, *benchmarkGames); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case modeGUI:
		gui := newGomokuApp(*aiMode, *modelPath)
		gui.window.ShowAndRun()
	default:
		fmt.Fprintf(os.Stderr, "unknown mode %q; expected %q or %q\n", *mode, modeGUI, modeTrain)
		os.Exit(1)
	}
}

func runTraining(modelPath string, episodes int, opponentDepth int, checkpointEvery int, evalGames int) error {
	cfg := rl.DefaultTrainerConfig()
	cfg.BoardSize = boardSize
	cfg.Episodes = episodes
	cfg.ScheduleEpisodes = episodes
	cfg.OpponentDepth = opponentDepth
	cfg.TeacherDepth = max(2, opponentDepth)
	if checkpointEvery <= 0 {
		checkpointEvery = episodes
	}
	if evalGames <= 0 {
		evalGames = 200
	}

	agent, err := rl.Load(modelPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			agent = rl.NewAgent()
		} else {
			return err
		}
	}

	bestPath := bestModelPath(modelPath)
	bestAgent := agent.Clone()
	bestEvalStats := rl.Stats{}
	bestEvalWinRate := -1.0
	totalStats := rl.Stats{Episodes: episodes}
	evalCfg := cfg
	evalCfg.MixedCurriculum = true

	if existingBest, err := rl.Load(bestPath); err == nil {
		existingBestStats := rl.Evaluate(existingBest, evalCfg, evalGames)
		bestAgent = existingBest
		bestEvalStats = existingBestStats
		bestEvalWinRate = existingBestStats.WinRate()
	}

	completed := 0
	for completed < episodes {
		chunkEpisodes := min(checkpointEvery, episodes-completed)
		chunkCfg := cfg
		chunkCfg.Episodes = chunkEpisodes
		chunkCfg.EpisodeOffset = completed
		chunkCfg.ScheduleEpisodes = episodes
		chunkStats := rl.Train(agent, chunkCfg)
		completed += chunkEpisodes

		totalStats.Wins += chunkStats.Wins
		totalStats.Losses += chunkStats.Losses
		totalStats.Draws += chunkStats.Draws
		totalStats.FinalEpsilon = chunkStats.FinalEpsilon

		if err := agent.Save(modelPath); err != nil {
			return err
		}

		evalStats := rl.Evaluate(agent, evalCfg, evalGames)
		isBest := false
		if evalStats.TotalGames() > 0 && evalStats.WinRate() >= bestEvalWinRate {
			bestEvalWinRate = evalStats.WinRate()
			bestEvalStats = evalStats
			bestAgent = agent.Clone()
			if err := bestAgent.Save(bestPath); err != nil {
				return err
			}
			isBest = true
		}

		fmt.Printf(
			"checkpoint %d/%d: train=%s   eval=%s   latest=%s",
			completed,
			episodes,
			rl.Summary(chunkStats),
			rl.Summary(evalStats),
			modelPath,
		)
		if isBest {
			fmt.Printf("   best=%s", bestPath)
		}
		fmt.Println()
	}

	if err := agent.Save(modelPath); err != nil {
		return err
	}

	fmt.Printf("Saved RL model to %s\n", modelPath)
	fmt.Println("Overall:", rl.Summary(totalStats))
	if bestEvalWinRate >= 0 {
		fmt.Printf("Best checkpoint model: %s\n", bestPath)
		fmt.Println("Best checkpoint eval stats:", rl.Summary(bestEvalStats))
	}
	return nil
}

func runImitation(modelPath string, games int, teacherDepth int) error {
	cfg := rl.DefaultImitationConfig()
	cfg.BoardSize = boardSize
	cfg.Games = games
	cfg.TeacherDepth = teacherDepth

	agent, err := rl.Load(modelPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			agent = rl.NewAgent()
		} else {
			return err
		}
	}

	stats := rl.TrainByImitation(agent, cfg)
	if err := agent.Save(modelPath); err != nil {
		return err
	}

	bestPath := bestModelPath(modelPath)
	if err := agent.Save(bestPath); err != nil {
		return err
	}

	fmt.Printf("Saved imitation-trained model to %s\n", modelPath)
	fmt.Printf("Updated strongest model snapshot at %s\n", bestPath)
	fmt.Println(rl.ImitationSummary(stats))
	return nil
}

func runHardening(modelPath string, games int, teacherDepth int) error {
	cfg := rl.DefaultHardeningConfig()
	cfg.BoardSize = boardSize
	cfg.Games = games
	cfg.TeacherDepth = teacherDepth

	agent, err := rl.Load(modelPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			agent = rl.NewAgent()
		} else {
			return err
		}
	}

	stats := rl.TrainOnHardPositions(agent, cfg)
	if err := agent.Save(modelPath); err != nil {
		return err
	}

	bestPath := bestModelPath(modelPath)
	if err := agent.Save(bestPath); err != nil {
		return err
	}

	fmt.Printf("Saved hardened RL model to %s\n", modelPath)
	fmt.Printf("Updated strongest model snapshot at %s\n", bestPath)
	fmt.Println(rl.HardeningSummary(stats))
	return nil
}

func runOptimize(modelPath string, cycles int, episodes int, opponentDepth int, checkpointEvery int, evalGames int, imitationGames int, hardeningGames int, sparringGames int, teacherDepth int, benchmarkGames int) error {
	if cycles <= 0 {
		return fmt.Errorf("cycles must be greater than 0")
	}
	if benchmarkGames <= 0 {
		return fmt.Errorf("games must be greater than 0")
	}

	workingAgent, err := rl.Load(modelPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			workingAgent = rl.NewAgent()
		} else {
			return err
		}
	}

	bestPath := bestModelPath(modelPath)
	bestAgent := workingAgent.Clone()
	bestStats := benchmarkAgentAgainstAlpha(bestAgent, benchmarkGames, 99, false)
	if savedBest, err := rl.Load(bestPath); err == nil {
		savedBestStats := benchmarkAgentAgainstAlpha(savedBest, benchmarkGames, 99, false)
		if betterBenchmark(savedBestStats, bestStats) {
			bestAgent = savedBest
			bestStats = savedBestStats
		}
	}

	fmt.Printf("Initial benchmark: %s\n", benchmarkSummary(bestStats))

	trainCfg := rl.DefaultTrainerConfig()
	trainCfg.BoardSize = boardSize
	trainCfg.Episodes = episodes
	trainCfg.ScheduleEpisodes = episodes
	trainCfg.OpponentDepth = opponentDepth
	trainCfg.TeacherDepth = max(2, opponentDepth)
	trainCfg.MixedCurriculum = true

	evalCfg := trainCfg
	evalCfg.MixedCurriculum = true

	imitCfg := rl.DefaultImitationConfig()
	imitCfg.BoardSize = boardSize
	imitCfg.Games = imitationGames
	imitCfg.TeacherDepth = teacherDepth

	hardenCfg := rl.DefaultHardeningConfig()
	hardenCfg.BoardSize = boardSize
	hardenCfg.Games = hardeningGames
	hardenCfg.TeacherDepth = teacherDepth

	sparCfg := rl.DefaultSparringConfig()
	sparCfg.BoardSize = boardSize
	sparCfg.Games = sparringGames
	sparCfg.TeacherDepth = teacherDepth

	for cycle := 1; cycle <= cycles; cycle++ {
		if imitationGames > 0 {
			imitStats := rl.TrainByImitation(workingAgent, imitCfg)
			fmt.Printf("cycle %d imitation: %s\n", cycle, rl.ImitationSummary(imitStats))
		}

		if episodes > 0 {
			if checkpointEvery <= 0 {
				checkpointEvery = episodes
			}

			completed := 0
			for completed < episodes {
				chunkEpisodes := min(checkpointEvery, episodes-completed)
				chunkCfg := trainCfg
				chunkCfg.Episodes = chunkEpisodes
				chunkCfg.EpisodeOffset = completed
				chunkCfg.ScheduleEpisodes = episodes
				chunkStats := rl.Train(workingAgent, chunkCfg)
				completed += chunkEpisodes
				fmt.Printf("cycle %d train %d/%d: %s\n", cycle, completed, episodes, rl.Summary(chunkStats))
			}
		}

		if hardeningGames > 0 {
			hardeningStats := rl.TrainOnHardPositions(workingAgent, hardenCfg)
			fmt.Printf("cycle %d harden: %s\n", cycle, rl.HardeningSummary(hardeningStats))
		}

		if sparringGames > 0 {
			sparringStats := rl.TrainBySparring(workingAgent, sparCfg)
			fmt.Printf("cycle %d sparring: %s\n", cycle, rl.SparringSummary(sparringStats))
		}

		if err := workingAgent.Save(modelPath); err != nil {
			return err
		}

		evalStats := rl.Stats{}
		if evalGames > 0 {
			evalStats = rl.Evaluate(workingAgent, evalCfg, evalGames)
		}
		cycleStats := benchmarkAgentAgainstAlpha(workingAgent, benchmarkGames, 99, true)

		fmt.Printf("cycle %d eval: %s\n", cycle, rl.Summary(evalStats))
		fmt.Printf("cycle %d benchmark: %s\n", cycle, benchmarkSummary(cycleStats))

		if betterBenchmark(cycleStats, bestStats) {
			bestStats = cycleStats
			bestAgent = workingAgent.Clone()
			if err := bestAgent.Save(bestPath); err != nil {
				return err
			}
			fmt.Printf("cycle %d promoted new strongest RL model at %s\n", cycle, bestPath)
		}
	}

	if err := workingAgent.Save(modelPath); err != nil {
		return err
	}
	if err := bestAgent.Save(bestPath); err != nil {
		return err
	}

	fmt.Printf("Latest RL model: %s\n", modelPath)
	fmt.Printf("Strongest RL model: %s\n", bestPath)
	fmt.Printf("Best benchmark: %s\n", benchmarkSummary(bestStats))
	return nil
}

func runBenchmark(modelPath string, games int) error {
	if games <= 0 {
		return fmt.Errorf("games must be greater than 0")
	}

	loadPath := modelPath
	if modelPath == defaultModelPath {
		loadPath = bestModelPath(defaultModelPath)
	}

	agent, err := rl.Load(loadPath)
	if err != nil {
		if loadPath != modelPath {
			agent, err = rl.Load(modelPath)
		}
		if err != nil {
			return err
		}
		loadPath = modelPath
	}

	stats := benchmarkAgentAgainstAlpha(agent, games, 99, true)

	alphaRate := 100 * float64(stats.AlphaBetaWins) / float64(stats.Games)
	rlRate := 100 * float64(stats.RLWins) / float64(stats.Games)
	drawRate := 100 * float64(stats.Draws) / float64(stats.Games)

	fmt.Printf("Benchmark model: %s\n", loadPath)
	fmt.Printf("Games: %d\n", stats.Games)
	fmt.Printf("Alpha-Beta wins: %d (%.1f%%)\n", stats.AlphaBetaWins, alphaRate)
	fmt.Printf("%s wins: %d (%.1f%%)\n", aiRL, stats.RLWins, rlRate)
	fmt.Printf("Draws: %d (%.1f%%)\n", stats.Draws, drawRate)
	fmt.Printf("First player split: Alpha-Beta=%d %s=%d\n", stats.AlphaBetaFirst, aiRL, stats.RLFirst)
	return nil
}

func benchmarkAgentAgainstAlpha(agent *rl.Agent, games int, seed int64, reportProgress bool) benchmarkStats {
	stats := benchmarkStats{Games: games}
	rng := rand.New(rand.NewSource(seed))

	for gameIndex := 0; gameIndex < games; gameIndex++ {
		if reportProgress {
			fmt.Printf("benchmark game %d/%d\n", gameIndex+1, games)
		}
		board := game.FullBoard(boardSize)
		alphaPlayer := game.P1
		if gameIndex%2 == 1 {
			alphaPlayer = game.P2
		}

		if alphaPlayer == game.P1 {
			stats.AlphaBetaFirst++
		} else {
			stats.RLFirst++
		}

		current := game.P1
		openingPlies := 0
		for !board.Full() {
			var move game.Move
			if openingPlies < 4 {
				move = chooseBenchmarkOpeningMove(board, current, rng)
			} else {
				move = chooseBenchmarkMove(board, current, agent, alphaPlayer)
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
				if current == alphaPlayer {
					stats.AlphaBetaWins++
				} else {
					stats.RLWins++
				}
				break
			}

			if board.Full() {
				stats.Draws++
				break
			}

			current = current.Other()
			openingPlies++
		}
	}

	return stats
}

func benchmarkWinRate(stats benchmarkStats) float64 {
	if stats.Games == 0 {
		return 0
	}
	return float64(stats.RLWins) / float64(stats.Games)
}

func benchmarkSummary(stats benchmarkStats) string {
	if stats.Games == 0 {
		return "games=0"
	}
	return fmt.Sprintf(
		"games=%d alpha-beta=%d rl=%d draws=%d rl-win-rate=%.1f%%",
		stats.Games,
		stats.AlphaBetaWins,
		stats.RLWins,
		stats.Draws,
		100*benchmarkWinRate(stats),
	)
}

func betterBenchmark(candidate, incumbent benchmarkStats) bool {
	if candidate.Games == 0 {
		return false
	}
	if candidate.RLWins != incumbent.RLWins {
		return candidate.RLWins > incumbent.RLWins
	}
	if candidate.AlphaBetaWins != incumbent.AlphaBetaWins {
		return candidate.AlphaBetaWins < incumbent.AlphaBetaWins
	}
	return candidate.Draws > incumbent.Draws
}

func chooseBenchmarkMove(board *game.Board, player game.Player, agent *rl.Agent, alphaPlayer game.Player) game.Move {
	if player == alphaPlayer {
		return board.BestMove(player, searchDepth)
	}
	if agent == nil {
		return board.BestMove(player, searchDepth)
	}
	return agent.BestMoveWithSearch(board, player, rlSearchDepth)
}

func chooseBenchmarkOpeningMove(board *game.Board, player game.Player, rng *rand.Rand) game.Move {
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
		center := float64(boardSize-1) / 2
		dist := absFloat(float64(move.Row)-center) + absFloat(float64(move.Col)-center)
		score -= dist * 50
		score += rng.Float64() * 50
		scored = append(scored, scoredMove{move: move, score: score})
	}

	if len(scored) == 0 {
		return moves[rng.Intn(len(moves))]
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})
	top := min(3, len(scored))
	return scored[rng.Intn(top)].move
}

func bestModelPath(modelPath string) string {
	ext := filepath.Ext(modelPath)
	base := modelPath[:len(modelPath)-len(ext)]
	if ext == "" {
		return modelPath + ".best"
	}
	return base + ".best" + ext
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func absFloat(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func newGomokuApp(aiMode string, modelPath string) *gomokuApp {
	a := app.New()
	w := a.NewWindow("Gomoku")
	w.Resize(fyne.NewSize(1120, 860))

	gui := &gomokuApp{
		board:       game.FullBoard(boardSize),
		humanPlayer: game.P1,
		aiPlayer:    game.P2,
		current:     game.P1,
		status:      "Your turn. Click any empty cell to place X.",
		modelPath:   modelPath,
		selectedAI:  normalizeAIMode(aiMode),
		matchMode:   modeForAI(normalizeAIMode(aiMode)),
		app:         a,
		window:      w,
		rng:         rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	if modelPath == defaultModelPath {
		gui.modelPath = bestModelPath(defaultModelPath)
	}

	gui.loadRLModel()
	gui.buildUI()
	gui.refreshBoard()

	return gui
}

func (g *gomokuApp) buildUI() {
	title := widget.NewRichTextFromMarkdown("# Gomoku")
	subtitle := widget.NewLabel("Play Gomoku against either the alpha-beta search bot or a trained reinforcement learning player.")
	subtitle.Wrapping = fyne.TextWrapWord

	g.statusLabel = widget.NewLabel(g.status)
	g.statusLabel.Wrapping = fyne.TextWrapWord

	g.infoLabel = widget.NewLabel("")
	g.lastMoveText = widget.NewLabel("Last move: none")

	restartButton := widget.NewButton("Restart Game", func() {
		g.resetGame()
	})

	g.aiSelect = widget.NewSelect([]string{aiAlphaBeta, aiRL}, nil)
	g.aiSelect.SetSelected(g.selectedAI)
	g.aiSelect.OnChanged = func(value string) {
		if g.syncingUI {
			return
		}
		g.selectedAI = normalizeAIMode(value)
		if g.matchMode != modeABVsRL {
			g.matchMode = modeForAI(g.selectedAI)
			g.setMatchSelection(g.matchMode)
		}
		g.refreshMeta()
		g.resetGame()
	}

	g.matchSelect = widget.NewSelect([]string{modeHumanVsAB, modeHumanVsRL, modeABVsRL}, nil)
	g.matchSelect.SetSelected(g.matchMode)
	g.matchSelect.OnChanged = func(value string) {
		if g.syncingUI {
			return
		}
		g.matchMode = normalizeMatchMode(value)
		if g.matchMode == modeHumanVsAB {
			g.selectedAI = aiAlphaBeta
			g.setAISelection(aiAlphaBeta)
		}
		if g.matchMode == modeHumanVsRL {
			g.selectedAI = aiRL
			g.setAISelection(aiRL)
		}
		g.refreshMeta()
		g.resetGame()
	}

	g.stepButton = widget.NewButton("Play Next AI Move", func() {
		g.advanceAIMatch(1)
	})
	g.autoButton = widget.NewButton("Run AI Match", func() {
		g.advanceAIMatch(boardSize * boardSize)
	})

	header := container.NewVBox(
		title,
		subtitle,
		container.NewHBox(
			restartButton,
			widget.NewLabel("Mode:"),
			g.matchSelect,
			widget.NewLabel("Opponent:"),
			g.aiSelect,
			g.stepButton,
			g.autoButton,
			layout.NewSpacer(),
			g.infoLabel,
		),
	)

	boardGrid := container.NewGridWithColumns(boardSize + 1)
	g.buttons = make([][]*widget.Button, boardSize)
	boardGrid.Add(widget.NewLabel(""))
	for col := 0; col < boardSize; col++ {
		label := widget.NewLabel(fmt.Sprintf("%2d", col))
		label.Alignment = fyne.TextAlignCenter
		boardGrid.Add(label)
	}

	for row := 0; row < boardSize; row++ {
		rowLabel := widget.NewLabel(fmt.Sprintf("%2d", row))
		rowLabel.Alignment = fyne.TextAlignCenter
		boardGrid.Add(rowLabel)

		g.buttons[row] = make([]*widget.Button, boardSize)
		for col := 0; col < boardSize; col++ {
			r := row
			c := col
			button := widget.NewButton("", func() {
				g.playHumanMove(r, c)
			})
			button.Importance = widget.LowImportance
			g.buttons[row][col] = button
			boardGrid.Add(button)
		}
	}

	sidebar := container.NewVBox(
		widget.NewCard("Status", "", g.statusLabel),
		widget.NewCard("Turn", "", g.lastMoveText),
		widget.NewCard("Opponents", "", widget.NewLabel(
			"Alpha-Beta uses the existing search player.\nReinforcement Learning uses the saved model at "+g.modelPath+".",
		)),
		widget.NewCard("Training", "", widget.NewLabel(
			"Improve the RL player with:\ngo run . -mode optimize -cycles 4 -games 40",
		)),
	)

	boardCard := widget.NewCard("Board", "Coordinates make it easier to track lines and moves.", container.NewPadded(boardGrid))

	content := container.NewBorder(
		header,
		nil,
		nil,
		sidebar,
		boardCard,
	)

	g.window.SetContent(content)
	g.refreshMeta()
}

func (g *gomokuApp) resetGame() {
	g.board = game.FullBoard(boardSize)
	g.humanPlayer = game.P1
	g.aiPlayer = game.P2
	g.current = game.P1
	g.aiMatchPlies = 0
	g.openingPlies = 4 + g.rng.Intn(4)
	switch g.matchMode {
	case modeABVsRL:
		if g.rng.Intn(2) == 0 {
			g.current = game.P1
		} else {
			g.current = game.P2
		}
		g.status = fmt.Sprintf("%s starts this match. Use the AI match controls to begin. Opening variety: %d plies.", g.labelForPlayer(g.current), g.openingPlies)
	case modeHumanVsRL:
		g.status = fmt.Sprintf("Your turn against %s. Click any empty cell to place X.", aiRL)
	default:
		g.status = fmt.Sprintf("Your turn against %s. Click any empty cell to place X.", aiAlphaBeta)
	}
	g.gameOver = false
	g.lastMove = nil
	g.refreshBoard()
}

func (g *gomokuApp) playHumanMove(row, col int) {
	if g.matchMode == modeABVsRL {
		g.showMessage("This mode is AI vs AI. Use Play Next AI Move or Run AI Match.")
		return
	}
	if g.gameOver {
		g.showMessage("The game is over. Press Restart Game to play again.")
		return
	}
	if g.current != g.humanPlayer {
		g.showMessage("Wait for the AI to finish its turn.")
		return
	}
	if err := g.board.Place(row, col, g.humanPlayer); err != nil {
		g.showMessage(err.Error())
		return
	}

	g.lastMove = &game.Move{Row: row, Col: col}
	if g.board.HasFive(row, col, g.humanPlayer) {
		g.status = fmt.Sprintf("Player %s wins!", g.humanPlayer.String())
		g.gameOver = true
		g.refreshBoard()
		return
	}
	if g.board.Full() {
		g.status = "Tie! Board is full!"
		g.gameOver = true
		g.refreshBoard()
		return
	}

	g.current = g.aiPlayer
	g.status = fmt.Sprintf("%s is thinking...", g.selectedAI)
	g.refreshBoard()

	move := g.chooseMoveFor(g.aiPlayer)
	if move.Row == -1 || move.Col == -1 {
		g.status = fmt.Sprintf("%s could not find a valid move.", g.selectedAI)
		g.gameOver = true
		g.refreshBoard()
		return
	}
	if err := g.board.Place(move.Row, move.Col, g.aiPlayer); err != nil {
		g.status = fmt.Sprintf("%s move failed: %v", g.selectedAI, err)
		g.gameOver = true
		g.refreshBoard()
		return
	}

	g.lastMove = &game.Move{Row: move.Row, Col: move.Col}
	if g.board.HasFive(move.Row, move.Col, g.aiPlayer) {
		g.status = fmt.Sprintf("%s (%s) wins!", g.selectedAI, g.aiPlayer.String())
		g.gameOver = true
		g.refreshBoard()
		return
	}
	if g.board.Full() {
		g.status = "Tie! Board is full!"
		g.gameOver = true
		g.refreshBoard()
		return
	}

	g.current = g.humanPlayer
	g.status = fmt.Sprintf("%s played %d, %d. Your turn.", g.selectedAI, move.Row, move.Col)
	g.refreshBoard()
}

func (g *gomokuApp) advanceAIMatch(maxMoves int) {
	if g.matchMode != modeABVsRL {
		g.showMessage("AI match controls are only available in Alpha-Beta vs Reinforcement Learning mode.")
		return
	}
	if g.gameOver {
		g.showMessage("The game is over. Press Restart Game to play again.")
		return
	}

	for i := 0; i < maxMoves && !g.gameOver; i++ {
		player := g.current
		label := g.labelForPlayer(player)
		g.status = fmt.Sprintf("%s is thinking...", label)
		g.refreshBoard()

		move := g.chooseMoveFor(player)
		if move.Row == -1 || move.Col == -1 {
			g.status = fmt.Sprintf("%s could not find a valid move.", label)
			g.gameOver = true
			break
		}
		if err := g.board.Place(move.Row, move.Col, player); err != nil {
			g.status = fmt.Sprintf("%s move failed: %v", label, err)
			g.gameOver = true
			break
		}

		g.lastMove = &game.Move{Row: move.Row, Col: move.Col}
		g.aiMatchPlies++
		if g.board.HasFive(move.Row, move.Col, player) {
			g.status = fmt.Sprintf("%s (%s) wins!", label, player.String())
			g.gameOver = true
			break
		}
		if g.board.Full() {
			g.status = "Tie! Board is full!"
			g.gameOver = true
			break
		}

		g.current = player.Other()
		g.status = fmt.Sprintf("%s played %d, %d. %s to move.", label, move.Row, move.Col, g.labelForPlayer(g.current))
	}

	g.refreshBoard()
}

func (g *gomokuApp) chooseMoveFor(player game.Player) game.Move {
	if g.matchMode == modeABVsRL {
		if g.aiMatchPlies < g.openingPlies {
			return g.chooseVariedOpeningMove(player)
		}
		if player == game.P1 {
			return g.board.BestMove(player, searchDepth)
		}
		if g.rlAgent != nil {
			return g.rlAgent.BestMoveWithSearch(g.board, player, rlSearchDepth)
		}
		return g.board.BestMove(player, searchDepth)
	}

	if g.selectedAI == aiRL && g.rlAgent != nil {
		return g.rlAgent.BestMoveWithSearch(g.board, player, rlSearchDepth)
	}
	return g.board.BestMove(player, searchDepth)
}

func (g *gomokuApp) chooseVariedOpeningMove(player game.Player) game.Move {
	moves := g.board.GenerateMoves()
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
		sim := g.board.Clone()
		if err := sim.Place(move.Row, move.Col, player); err != nil {
			continue
		}

		score := float64(sim.Evaluate(player))
		score += float64(sim.LongestLineAt(move.Row, move.Col, player)) * 2500
		score -= float64(len(sim.WinningMoves(player.Other()))) * 4000

		center := float64(boardSize-1) / 2
		dist := absFloat(float64(move.Row)-center) + absFloat(float64(move.Col)-center)
		score -= dist * 50
		score += g.rng.Float64() * 75

		scored = append(scored, scoredMove{move: move, score: score})
	}

	if len(scored) == 0 {
		return moves[g.rng.Intn(len(moves))]
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	window := min(6, len(scored))
	if g.aiMatchPlies <= 1 {
		window = min(10, len(scored))
	}

	weightedTotal := 0.0
	weights := make([]float64, window)
	bestScore := scored[0].score
	for i := 0; i < window; i++ {
		gap := bestScore - scored[i].score
		weight := 1.0 / (1.0 + absFloat(gap)/1500.0)
		if i == 0 {
			weight *= 1.15
		}
		if i >= 3 {
			weight *= 0.85
		}
		weights[i] = weight
		weightedTotal += weight
	}

	pick := g.rng.Float64() * weightedTotal
	running := 0.0
	for i := 0; i < window; i++ {
		running += weights[i]
		if pick <= running {
			return scored[i].move
		}
	}

	return scored[window-1].move
}

func (g *gomokuApp) refreshBoard() {
	for row := 0; row < boardSize; row++ {
		for col := 0; col < boardSize; col++ {
			cell := g.board.Get(row, col)
			button := g.buttons[row][col]
			button.SetText(g.buttonText(cell, row, col))
			button.Importance = g.buttonImportance(cell)

			if g.canTapCell(cell) {
				button.Enable()
			} else if cell != game.Empty {
				button.Enable()
			} else {
				button.Disable()
			}

			button.Refresh()
		}
	}

	g.statusLabel.SetText(g.status)
	if g.lastMove == nil {
		g.lastMoveText.SetText("Last move: none")
	} else {
		g.lastMoveText.SetText(fmt.Sprintf("Last move: %d, %d", g.lastMove.Row, g.lastMove.Col))
	}
	g.refreshMeta()
	g.window.Content().Refresh()
}

func (g *gomokuApp) refreshMeta() {
	info := fmt.Sprintf("Board: %dx%d   Alpha-Beta depth: %d   Mode: %s", boardSize, boardSize, searchDepth, g.matchMode)
	if g.matchMode == modeABVsRL {
		info += "   P1: Alpha-Beta   P2: Reinforcement Learning"
	} else {
		info += fmt.Sprintf("   Active AI: %s", g.selectedAI)
	}
	if g.selectedAI == aiRL || g.matchMode == modeABVsRL {
		if g.rlAgent != nil {
			info += "   RL model: loaded"
		} else {
			info += "   RL model: unavailable, using Alpha-Beta fallback"
		}
	}
	g.infoLabel.SetText(info)
	if g.stepButton != nil {
		if g.matchMode == modeABVsRL && !g.gameOver {
			g.stepButton.Enable()
			g.autoButton.Enable()
		} else {
			g.stepButton.Disable()
			g.autoButton.Disable()
		}
	}
}

func (g *gomokuApp) showMessage(message string) {
	dialog.ShowInformation("Gomoku", message, g.window)
}

func (g *gomokuApp) canTapCell(cell game.Player) bool {
	return g.matchMode != modeABVsRL && cell == game.Empty && !g.gameOver && g.current == g.humanPlayer
}

func (g *gomokuApp) buttonImportance(cell game.Player) widget.ButtonImportance {
	switch cell {
	case game.P1:
		return widget.DangerImportance
	case game.P2:
		return widget.HighImportance
	default:
		return widget.LowImportance
	}
}

func (g *gomokuApp) buttonText(cell game.Player, row, col int) string {
	switch cell {
	case game.P1:
		if g.isLastMove(row, col) {
			return "X*"
		}
		return "X"
	case game.P2:
		if g.isLastMove(row, col) {
			return "Y*"
		}
		return "Y"
	default:
		return "·"
	}
}

func (g *gomokuApp) isLastMove(row, col int) bool {
	return g.lastMove != nil && g.lastMove.Row == row && g.lastMove.Col == col
}

func (g *gomokuApp) loadRLModel() {
	agent, err := rl.Load(g.modelPath)
	if err != nil {
		if g.modelPath == bestModelPath(defaultModelPath) {
			fallbackAgent, fallbackErr := rl.Load(defaultModelPath)
			if fallbackErr == nil {
				g.rlModelError = nil
				g.rlAgent = fallbackAgent
				g.modelPath = defaultModelPath
				return
			}
		}
		g.rlModelError = err
		g.rlAgent = nil
		return
	}
	g.rlModelError = nil
	g.rlAgent = agent
}

func normalizeAIMode(value string) string {
	switch value {
	case aiRL:
		return aiRL
	default:
		return aiAlphaBeta
	}
}

func (g *gomokuApp) setAISelection(value string) {
	g.syncingUI = true
	g.aiSelect.SetSelected(value)
	g.syncingUI = false
}

func (g *gomokuApp) setMatchSelection(value string) {
	g.syncingUI = true
	g.matchSelect.SetSelected(value)
	g.syncingUI = false
}

func normalizeMatchMode(value string) string {
	switch value {
	case modeHumanVsRL:
		return modeHumanVsRL
	case modeABVsRL:
		return modeABVsRL
	default:
		return modeHumanVsAB
	}
}

func modeForAI(ai string) string {
	if ai == aiRL {
		return modeHumanVsRL
	}
	return modeHumanVsAB
}

func (g *gomokuApp) labelForPlayer(player game.Player) string {
	if g.matchMode == modeABVsRL {
		if player == game.P1 {
			return aiAlphaBeta
		}
		return aiRL
	}
	return g.selectedAI
}
