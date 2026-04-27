package game

// this function scores the board from the pov of the agent or aiplayer
// positive scores favor the agent
// negative scores favor the opponent or the real person.
func (b *Board) Evaluate(aiPlayer Player) int {
	opponent := aiPlayer.Other()
	score := 0

	// this gives some center board preference since center positions are usually stronger
	center := b.Size / 2
	for r := 0; r < b.Size; r++ {
		for c := 0; c < b.Size; c++ {
			if b.Cells[r][c] == aiPlayer {
				score += 5 - manhattanDist(r, c, center, center)
			} else if b.Cells[r][c] == opponent {
				score -= 5 - manhattanDist(r, c, center, center)
			}
		}
	}

	// this section evaluates the rows
	for r := 0; r < b.Size; r++ {
		line := make([]Player, b.Size)
		for c := 0; c < b.Size; c++ {
			line[c] = b.Cells[r][c]
		}
		score += evalLine(line, aiPlayer)
		score -= evalLine(line, opponent)
	}

	//this section evauluates the columns
	for c := 0; c < b.Size; c++ {
		line := make([]Player, b.Size)
		for r := 0; r < b.Size; r++ {
			line[r] = b.Cells[r][c]
		}
		score += evalLine(line, aiPlayer)
		score -= evalLine(line, opponent)
	}

	// this part evaluates the down right diagonals
	for startRow := 0; startRow < b.Size; startRow++ {
		line := []Player{}
		r, c := startRow, 0
		for b.InBounds(r, c) {
			line = append(line, b.Cells[r][c])
			r++
			c++
		}
		if len(line) >= 5 {
			score += evalLine(line, aiPlayer)
			score -= evalLine(line, opponent)
		}
	}
	for startCol := 1; startCol < b.Size; startCol++ {
		line := []Player{}
		r, c := 0, startCol
		for b.InBounds(r, c) {
			line = append(line, b.Cells[r][c])
			r++
			c++
		}
		if len(line) >= 5 {
			score += evalLine(line, aiPlayer)
			score -= evalLine(line, opponent)
		}
	}

	// this part evaluates the up right diagonals
	for startRow := 0; startRow < b.Size; startRow++ {
		line := []Player{}
		r, c := startRow, 0
		for b.InBounds(r, c) {
			line = append(line, b.Cells[r][c])
			r--
			c++
		}
		if len(line) >= 5 {
			score += evalLine(line, aiPlayer)
			score -= evalLine(line, opponent)
		}
	}
	for startCol := 1; startCol < b.Size; startCol++ {
		line := []Player{}
		r, c := b.Size-1, startCol
		for b.InBounds(r, c) {
			line = append(line, b.Cells[r][c])
			r--
			c++
		}
		if len(line) >= 5 {
			score += evalLine(line, aiPlayer)
			score -= evalLine(line, opponent)
		}
	}
	return score
}

// this function checks one row/column/diagonal for patterns that belong to the player
func evalLine(line []Player, player Player) int {
	score := 0
	n := len(line)

	for i := 0; i < n; i++ {
		if line[i] != player {
			continue
		}

		// this counts the conseutives pieces (X)
		j := i
		for j < n && line[j] == player {
			j++
		}
		runLen := j - i

		leftOpen := i-1 >= 0 && line[i-1] == Empty
		rightOpen := j < n && line[j] == Empty
		openEnds := 0
		if leftOpen {
			openEnds++
		}
		if rightOpen {
			openEnds++
		}

		score += scoreRun(runLen, openEnds)

		// this skips to the end of this specific run
		i = j - 1
	}
	return score
}

// scoreRun returns a score for a run of pieces with some number of open ends.
func scoreRun(runLen int, openEnds int) int {
	if openEnds == 0 && runLen < 5 {
		return 0
	}

	switch runLen {
	case 5:
		return 1_000_000
	case 4:
		if openEnds == 2 {
			return 100_000
		}
		if openEnds == 1 {
			return 10_000
		}
	case 3:
		if openEnds == 2 {
			return 5_000
		}
		if openEnds == 1 {
			return 500
		}
	case 2:
		if openEnds == 2 {
			return 200
		}
		if openEnds == 1 {
			return 50
		}
	case 1:
		if openEnds == 2 {
			return 10
		}
		if openEnds == 1 {
			return 1
		}
	}

	if runLen > 5 {
		return 1_000_000
	}

	return 0
}

// this manhattandistance function helps to give some bias to center placement for the agent
func manhattanDist(r1, c1, r2, c2 int) int {
	dr := r1 - r2
	if dr < 0 {
		dr = -dr
	}
	dc := c1 - c2
	if dc < 0 {
		dc = -dc
	}
	return dr + dc
}
