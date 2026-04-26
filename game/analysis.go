package game

// LongestLineAt returns the longest contiguous run that includes r,c for player p.
func (b *Board) LongestLineAt(r, c int, p Player) int {
	if p == Empty || !b.InBounds(r, c) || b.Get(r, c) != p {
		return 0
	}

	best := 1
	directions := [][2]int{
		{0, 1},
		{1, 0},
		{1, 1},
		{-1, 1},
	}

	for _, d := range directions {
		dr, dc := d[0], d[1]
		total := 1 + b.countDir(r, c, dr, dc, p) + b.countDir(r, c, -dr, -dc, p)
		if total > best {
			best = total
		}
	}

	return best
}

// WinningMoves returns every legal move that would win immediately for player p.
func (b *Board) WinningMoves(p Player) []Move {
	moves := b.GenerateMoves()
	wins := make([]Move, 0, len(moves))

	for _, move := range moves {
		if err := b.Place(move.Row, move.Col, p); err != nil {
			continue
		}
		if b.HasFive(move.Row, move.Col, p) {
			wins = append(wins, move)
		}
		b.Remove(move.Row, move.Col)
	}

	return wins
}

// OccupiedNeighbors counts occupied cells within distance steps of r,c.
func (b *Board) OccupiedNeighbors(r, c, distance int) int {
	count := 0
	for dr := -distance; dr <= distance; dr++ {
		for dc := -distance; dc <= distance; dc++ {
			if dr == 0 && dc == 0 {
				continue
			}

			rr := r + dr
			cc := c + dc
			if b.InBounds(rr, cc) && b.Cells[rr][cc] != Empty {
				count++
			}
		}
	}

	return count
}

// MaxLine returns the longest contiguous line currently on the board for player p.
func (b *Board) MaxLine(p Player) int {
	best := 0
	for r := 0; r < b.Size; r++ {
		for c := 0; c < b.Size; c++ {
			if b.Get(r, c) != p {
				continue
			}
			line := b.LongestLineAt(r, c, p)
			if line > best {
				best = line
			}
		}
	}
	return best
}

// Count returns the number of stones currently on the board for player p.
func (b *Board) Count(p Player) int {
	total := 0
	for r := 0; r < b.Size; r++ {
		for c := 0; c < b.Size; c++ {
			if b.Cells[r][c] == p {
				total++
			}
		}
	}
	return total
}
