package game

// this struct represents one move on the board.
type Move struct {
	Row int
	Col int
}

// this returns some amount of possible moves for the current dynamics of the board.
// the strat here is that if the board is empty then it returns the center position.
// if not empty then it returns empty spaces that are within a short distance of at least one occupied space
// this is more optimal than checking every empty space on the board,
// since moves far from existing pieces aren't strong.
func (b *Board) GenerateMoves() []Move {
	if b.isEmptyBoard() {
		center := b.Size / 2
		return []Move{{Row: center, Col: center}}
	}

	moves := []Move{}

	for r := 0; r < b.Size; r++ {
		for c := 0; c < b.Size; c++ {
			if b.Cells[r][c] != Empty {
				continue
			}

			// only keeps the empty cells that are near an existing peice.
			if b.HasNeighbor(r, c, 2) {
				moves = append(moves, Move{Row: r, Col: c})
			}
		}
	}
	return moves
}

// this checks if the specific cell (r, c) has at least one occupied cell within it's near distance.
// so a distance of 2 means that it would search the square from (r-2, c-2) to (r+2, c+2)
func (b *Board) HasNeighbor(r, c, distance int) bool {
	for dr := -distance; dr <= distance; dr++ {
		for dc := -distance; dc <= distance; dc++ {
			//skips the actual cell itself
			if dr == 0 && dc == 0 {
				continue
			}

			rr := r + dr
			cc := c + dc

			if b.InBounds(rr, cc) && b.Cells[rr][cc] != Empty {
				return true
			}
		}
	}
	return false
}

// this returns true if there aren't any stones on the board.
func (b *Board) isEmptyBoard() bool {
	for r := 0; r < b.Size; r++ {
		for c := 0; c < b.Size; c++ {
			if b.Cells[r][c] != Empty {
				return false
			}
		}
	}
	return true
}
