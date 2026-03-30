- Overview 
This project is a command line version of the board game Gomoku, written in Go.
The game is played on a 15*15 grid where two players take turns putting down their pieces
with the goal being to get 5 in a row vertically, horizontaly, or diagonally.

The project started as a PvP game and was later extended to include an alpha beta (AI) player
that uses alpha beta pruning and a heuristic evaluation function to make decisions.

Win detection
- the game checks for a win by looking at the most recent move and scanning in 4 directions.
- This approach allows for efficient searching and avoids having to scan the whole board.

AI Implementation
- The evaluation function scores the board by identifying pattenrs like:
- 5 in a row
- open a closed sequences of 2, 3, and 4 pieces
- center control preference.
- Positive scores favor the AI, while negative scores favor the opponent.

- Alpha beta pruning
The AI uses a depth limited minimax search with alpha beta pruning to lower the number
of explored positions. Currently I set the search depth to 3,as this seems to be a good
middle ground for performance. This allows the program to evaulate moves efficiently without having
to search through the entire game tree.

Future Improvments/Goals
- GUI
- possibly stronger evaluation function
- Added Reinforcement Learning player to assess differences in perfromance to AI player.
- possibly added difficulty levels.

- Inspiration
This project was inspired by my interest in game Ai and algorithm design. Gomoku is similar to 
Connect 4, which is a game that I enjoy playing. The rules are simple for Gomoku, but it has 
a complex environment which made it a good challenge and a good problem for exploring
search algorithms and heuristic evaluation.
