package game

import (
	"errors"
	"fmt"
	"strings"
)

// Board represents the game board
type Board struct {
	Size  int
	Cells [][]Player
}

// FullBoard makes a new square board with all of the cells being empty
func FullBoard(size int) *Board {
	cells := make([][]Player, size)
	for r := 0; r < size; r++ {
		cells[r] = make([]Player, size)
	}
	return &Board{Size: size, Cells: cells}
}

// this checks if a coordinate that is put is actually in the board.
func (b *Board) InBounds(r, c int) bool {
	return r >= 0 && r < b.Size && c >= 0 && c < b.Size
}

// Get returns the player at the coordinate given.
func (b *Board) Get(r, c int) Player {
	if !b.InBounds(r, c) {
		return Empty
	}
	return b.Cells[r][c]
}

// Place tries to put a player's piece at the row, col
// Returns an error if the move isn't valid.
func (b *Board) Place(r, c int, p Player) error {
	if !b.InBounds(r, c) {
		return fmt.Errorf("this is out of bounds: row/col must be 0..%d", b.Size-1)
	}
	if p != P1 && p != P2 {
		return errors.New("invalid player")
	}
	if b.Cells[r][c] != Empty {
		return errors.New("cell is already in use")
	}
	b.Cells[r][c] = p
	return nil
}

// Full checks if the board has no empty spaces left.
func (b *Board) Full() bool {
	for r := 0; r < b.Size; r++ {
		for c := 0; c < b.Size; c++ {
			if b.Cells[r][c] == Empty {
				return false
			}
		}
	}
	return true
}

// Render builds the string representation of the board that is shown in the terminal
func (b *Board) Render() string {
	var sb strings.Builder

	// These are the column headers
	sb.WriteString("   ")
	for c := 0; c < b.Size; c++ {
		sb.WriteString(fmt.Sprintf("%3d", c))
	}
	sb.WriteString("\n")

	// These are the rows
	for r := 0; r < b.Size; r++ {
		sb.WriteString(fmt.Sprintf("%3d ", r))
		for c := 0; c < b.Size; c++ {
			sb.WriteString(fmt.Sprintf("%3s", b.Cells[r][c].String()))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}
