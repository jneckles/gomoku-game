package main

import (
	"fmt"
	"gomoku/game"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"
)

const boardSize = 15
const searchDepth = 3

type gomokuApp struct {
	board       *game.Board
	humanPlayer game.Player
	aiPlayer    game.Player
	current     game.Player
	status      string
	gameOver    bool
	lastMove    *game.Move

	app          fyne.App
	window       fyne.Window
	statusLabel  *widget.Label
	infoLabel    *widget.Label
	buttons      [][]*widget.Button
	lastMoveText *widget.Label
}

func main() {
	gui := newGomokuApp()
	gui.window.ShowAndRun()
}

func newGomokuApp() *gomokuApp {
	a := app.New()
	w := a.NewWindow("Gomoku")
	w.Resize(fyne.NewSize(980, 860))

	gui := &gomokuApp{
		board:       game.FullBoard(boardSize),
		humanPlayer: game.P1,
		aiPlayer:    game.P2,
		current:     game.P1,
		status:      "Your turn. Click any empty cell to place X.",
		app:         a,
		window:      w,
	}

	gui.buildUI()
	gui.refreshBoard()

	return gui
}

func (g *gomokuApp) buildUI() {
	title := widget.NewRichTextFromMarkdown("# Gomoku")
	subtitle := widget.NewLabel("You are X. The AI is Y. Get five in a row to win.")
	subtitle.Wrapping = fyne.TextWrapWord

	g.statusLabel = widget.NewLabel(g.status)
	g.statusLabel.Wrapping = fyne.TextWrapWord

	g.infoLabel = widget.NewLabel(fmt.Sprintf("Board: %dx%d   Search depth: %d", boardSize, boardSize, searchDepth))
	g.lastMoveText = widget.NewLabel("Last move: none")

	restartButton := widget.NewButton("Restart Game", func() {
		g.resetGame()
	})

	header := container.NewVBox(
		title,
		subtitle,
		container.NewHBox(restartButton, layout.NewSpacer(), g.infoLabel),
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
		widget.NewCard("How To Play", "", widget.NewLabel("Click an empty square to place your piece. The AI responds immediately using the same search logic as before.")),
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
}

func (g *gomokuApp) resetGame() {
	g.board = game.FullBoard(boardSize)
	g.humanPlayer = game.P1
	g.aiPlayer = game.P2
	g.current = game.P1
	g.status = "Your turn. Click any empty cell to place X."
	g.gameOver = false
	g.lastMove = nil
	g.refreshBoard()
}

func (g *gomokuApp) playHumanMove(row, col int) {
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
	g.status = "AI is thinking..."
	g.refreshBoard()

	move := g.board.BestMove(g.aiPlayer, searchDepth)
	if move.Row == -1 || move.Col == -1 {
		g.status = "AI could not find a valid move."
		g.gameOver = true
		g.refreshBoard()
		return
	}
	if err := g.board.Place(move.Row, move.Col, g.aiPlayer); err != nil {
		g.status = fmt.Sprintf("AI move failed: %v", err)
		g.gameOver = true
		g.refreshBoard()
		return
	}

	g.lastMove = &game.Move{Row: move.Row, Col: move.Col}
	if g.board.HasFive(move.Row, move.Col, g.aiPlayer) {
		g.status = fmt.Sprintf("AI (%s) wins!", g.aiPlayer.String())
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
	g.status = fmt.Sprintf("AI played %d, %d. Your turn.", move.Row, move.Col)
	g.refreshBoard()
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
				// Keep occupied cells visually strong instead of graying them out.
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

	g.window.Content().Refresh()
}

func (g *gomokuApp) showMessage(message string) {
	dialog.ShowInformation("Gomoku", message, g.window)
}

func (g *gomokuApp) canTapCell(cell game.Player) bool {
	return cell == game.Empty && !g.gameOver && g.current == g.humanPlayer
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
