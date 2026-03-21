package main

import (
	"bufio"
	"fmt"
	"gomoku/game"
	"os"
	"os/exec"
	"strings"
)

// this just defines the dimension of the game board 15 * 15
const boardSize = 15

// this controls how far the AI looks ahead
// so far 3 seems to be a good spot where it doesn't run slow
const searchDepth = 3

func main() {
	// makea a new board with the size (15X15)
	b := game.FullBoard(boardSize)
	// reader is used here to get user input from stdin
	reader := bufio.NewReader(os.Stdin)

	// current just keeps track of whose turn it is.
	// human player is X, ai is Y.
	humanPlayer := game.P1
	aiPlayer := game.P2
	current := game.P1

	// prints the intro and then waits before starting
	printIntro(boardSize)
	fmt.Println("Press Enter to start :)")
	reader.ReadString('\n')

	// the game loop until a win, draw or quit.
	for {
		clearScreen()
		// renders board to the terminal

		fmt.Print(b.Render())

		if current == humanPlayer {
			// this prompts the current player to make move
			fmt.Printf("player %s, enter row col: ", current.String())

			//reads the move from the player
			row, col, quit, err := readMove(reader)

			// handles the quit command, ends game
			if quit {
				fmt.Println("Bye Bye.")
				return
			}
			// handles invalid input
			if err != nil {
				fmt.Println(err)
				fmt.Println("Press Enter to continue :)")
				reader.ReadString('\n')
				continue
			}

			// this attempts to place the move and if not accepted, makes the user continue.
			if err := b.Place(row, col, current); err != nil {
				fmt.Printf("Move not accepted: %v\n\n", err)
				fmt.Println("press Enter to continue...")
				reader.ReadString('\n')
				continue
			}

			// checks for the win conditon (5 in a row) and outputs win message.
			if b.HasFive(row, col, current) {
				clearScreen()
				fmt.Print(b.Render())
				fmt.Printf("Player %s wins!\n", current.String())
				return
			}

			// checks for a tie if the board is full and no 5 in a row.
			if b.Full() {
				clearScreen()
				fmt.Print(b.Render())
				fmt.Println("Tie! Board is full!")
				return
			}

			// switches turns between players
			current = current.Other()
			continue
			//fmt.Println()
		}
		//Ai's turn
		fmt.Printf("AI (%s) is deciding hmm... \n", aiPlayer.String())

		move := b.BestMove(aiPlayer, searchDepth)

		// just a safe check in case no valid move is found
		if move.Row == -1 || move.Col == -1 {
			fmt.Println("Ai could not find a valid move.")
			return
		}

		if err := b.Place(move.Row, move.Col, aiPlayer); err != nil {
			fmt.Printf("Ai move failed: %v\n", err)
			return
		}

		// shows what move the Ai played after it puts down a piece.
		clearScreen()
		fmt.Print(b.Render())
		fmt.Printf("Ai played: %d %d\n", move.Row, move.Col)

		// checks if the ai won
		if b.HasFive(move.Row, move.Col, aiPlayer) {
			fmt.Printf("Ai (%s) wins!\n", aiPlayer.String())
			return
		}

		//checks for a tie
		if b.Full() {
			fmt.Println("Tie! Board is full!")
			return
		}

		fmt.Println("Press Enter to continue...")
		reader.ReadString('\n')

		current = current.Other()

	}

}

// this function parses the user input
// it returns row, col, quit flag and an error if the input isn't valid.
func readMove(reader *bufio.Reader) (row int, col int, quit bool, err error) {
	line, _ := reader.ReadString('\n')
	line = strings.TrimSpace(line)

	// checks the quit commands
	if line == "q" || line == "quit" || line == "exit" {
		return 0, 0, true, nil
	}

	var r, c int
	n, scanErr := fmt.Sscanf(line, "%d %d", &r, &c)
	if scanErr != nil || n != 2 {
		return 0, 0, false, fmt.Errorf("invalid input. Example: 7 8")
	}

	return r, c, false, nil
}

// clearScreen clears the terminal using clear command for the system.
// This keeps the board in one place and doesn't stack the board after each move
func clearScreen() {
	cmd := exec.Command("clear")
	cmd.Stdout = os.Stdout
	_ = cmd.Run()
}

// These are the intro messages for the game
func printIntro(size int) {
	fmt.Printf("This is Gomoku on a %dx%d board.\n", size, size)
	fmt.Println("You are X. The AI is 0")
	fmt.Println("If you get 5 in a row then you win!")
	fmt.Println("Put in your move as: row col (ex: 6 8)")
	fmt.Println("Type q to quit during a turn")
	fmt.Println()
}
