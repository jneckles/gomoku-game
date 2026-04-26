package rl

import "gomoku/game"

const symmetryAugmentations = 4

func chooseSymmetries(rngIndex func(int) int) []int {
	if symmetryAugmentations <= 1 {
		return []int{0}
	}

	used := map[int]bool{0: true}
	out := []int{0}
	for len(out) < symmetryAugmentations && len(used) < 8 {
		next := rngIndex(8)
		if used[next] {
			continue
		}
		used[next] = true
		out = append(out, next)
	}
	return out
}

func transformBoardAndMove(board *game.Board, move game.Move, symmetry int) (*game.Board, game.Move) {
	size := board.Size
	transformed := game.FullBoard(size)

	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			tr, tc := transformCoord(r, c, size, symmetry)
			transformed.Cells[tr][tc] = board.Cells[r][c]
		}
	}

	mr, mc := transformCoord(move.Row, move.Col, size, symmetry)
	return transformed, game.Move{Row: mr, Col: mc}
}

func transformCoord(r, c, size, symmetry int) (int, int) {
	last := size - 1
	switch symmetry {
	case 0:
		return r, c
	case 1:
		return c, last - r
	case 2:
		return last - r, last - c
	case 3:
		return last - c, r
	case 4:
		return r, last - c
	case 5:
		return last - r, c
	case 6:
		return c, r
	case 7:
		return last - c, last - r
	default:
		return r, c
	}
}
