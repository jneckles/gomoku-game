package game

// Player is the state of a cell
//0 is empty
// 1 is Player 1
// 2 is Player 2

type Player int

const (
	Empty Player = iota
	P1
	P2
)

// String returns the symbol for the player that's printed on the board
func (p Player) String() string {
	switch p {
	case P1:
		return "X"
	case P2:
		return "Y"
	default:
		return "."
	}
}

// Other switches the current player
// If P1 then returns P2 and vice versa.
func (p Player) Other() Player {
	if p == P1 {
		return P2
	}
	if p == P2 {
		return P1
	}
	return Empty
}
