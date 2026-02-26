package game

// HasFive checks if the player placing the stone at a row col pair gets them to 5 in a row.
func (b *Board) HasFive(r, c int, p Player) bool {
	if p == Empty {
		return false
	}

	// These are the directions that have to get checked
	directions := [][2]int{
		{0, 1},  // horizontal
		{1, 0},  //vertical
		{1, 1},  // diagonal down right
		{-1, 1}, //diagonal up right
	}

	// counts the pieces in both directions
	for _, d := range directions {
		dr, dc := d[0], d[1]
		total := 1 + b.countDir(r, c, dr, dc, p) + b.countDir(r, c, -dr, -dc, p)
		if total >= 5 {
			return true
		}
	}
	return false
}

// countDir counts the stones in one direction
func (b *Board) countDir(row, col, dr, dc int, p Player) int {
	count := 0
	rr := row + dr
	cc := col + dc
	for b.InBounds(rr, cc) && b.Get(rr, cc) == p {
		count++
		rr += dr
		cc += dc
	}
	return count

}
