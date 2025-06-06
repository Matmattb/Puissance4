import random
import time
import copy
import numpy as np

# Constants
ROWS = 6
COLS = 12
AI_PLAYER = 1  # AI
HUMAN_PLAYER = -1  # Human
EMPTY = 0
MAX_DEPTH = 6  # Depth for Minimax (tune for 10-second limit)

def is_valid_move(board, col):
    """Check if a move in the given column is valid."""
    return 0 <= col < COLS and board[0][col] == EMPTY

def get_valid_moves(board):
    """Return a list of valid column indices for moves."""
    return [col for col in range(COLS) if is_valid_move(board, col)]

def drop_piece(board, col, player):
    """Drop a piece in the specified column for the given player."""
    new_board = copy.deepcopy(board)
    for row in range(ROWS - 1, -1, -1):
        if new_board[row][col] == EMPTY:
            new_board[row][col] = player
            return new_board, row
    return new_board, -1  # Should not reach here if move is valid

def check_win(board, player):
    """Check if the given player has won (four in a row)."""
    # Horizontal
    for row in range(ROWS):
        for col in range(COLS - 3):
            if all(board[row][col + i] == player for i in range(4)):
                return True

    # Vertical
    for row in range(ROWS - 3):
        for col in range(COLS):
            if all(board[row + i][col] == player for i in range(4)):
                return True

    # Diagonal (positive slope)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if all(board[row + i][col + i] == player for i in range(4)):
                return True

    # Diagonal (negative slope)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            if all(board[row - i][col + i] == player for i in range(4)):
                return True

    return False

def is_board_full(board):
    """Check if the board is full (draw)."""
    return all(board[0][col] != EMPTY for col in range(COLS))

def Terminal_Test(board):
    """Check if the game is over (win or draw)."""
    return check_win(board, AI_PLAYER) or check_win(board, HUMAN_PLAYER) or is_board_full(board)

def Utility(board):
    """Return utility value for terminal state."""
    if check_win(board, AI_PLAYER):
        return 10000  # Increased to ensure win priority
    if check_win(board, HUMAN_PLAYER):
        return -10000
    return 0  # Draw

def check_immediate_threat(board, player, col):
    """Check if placing a piece in col creates a four-in-a-row for player."""
    temp_board, row = drop_piece(board, col, player)
    if row == -1:  # Invalid move
        return False
    return check_win(temp_board, player)

def prioritize_moves(board):
    """Order moves to check winning moves first, then blocking moves, then center columns."""
    valid_moves = get_valid_moves(board)
    winning_moves = []
    blocking_moves = []
    other_moves = []

    for col in valid_moves:
        # Check if AI can win
        if check_immediate_threat(board, AI_PLAYER, col):
            winning_moves.append(col)
        # Check if human can win (AI needs to block)
        elif check_immediate_threat(board, HUMAN_PLAYER, col):
            blocking_moves.append(col)
        else:
            other_moves.append(col)

    # Sort other moves by center preference
    center_columns = [5, 6, 4, 7, 3, 8]
    other_moves.sort(key=lambda x: center_columns.index(x) if x in center_columns else len(center_columns))

    return winning_moves + blocking_moves + other_moves

def heuristic(board):
    """Heuristic evaluation for non-terminal states."""
    score = 0
    weights = {3: 100, 2: 10, 1: 1}  # Weights for sequences

    def count_sequence(line, player):
        count = 0
        empty = 0
        seq_score = 0
        for cell in line:
            if cell == player:
                count += 1
            elif cell == EMPTY:
                empty += 1
            else:
                if count == 3 and empty >= 1:
                    seq_score += 1000 if player == AI_PLAYER else -1000
                elif count in weights and empty >= (4 - count):
                    seq_score += weights.get(count, 0)
                count = 0
                empty = 0
        if count == 3 and empty >= 1:
            seq_score += 1000 if player == AI_PLAYER else -1000
        elif count in weights and empty >= (4 - count):
            seq_score += weights.get(count, 0)
        return seq_score

    # Horizontal
    for row in range(ROWS):
        for col in range(COLS - 3):
            line = [board[row][col + i] for i in range(4)]
            score += count_sequence(line, AI_PLAYER)
            score -= count_sequence(line, HUMAN_PLAYER)

    # Vertical
    for col in range(COLS):
        for row in range(ROWS - 3):
            line = [board[row + i][col] for i in range(4)]
            score += count_sequence(line, AI_PLAYER)
            score -= count_sequence(line, HUMAN_PLAYER)

    # Diagonal (positive slope)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            line = [board[row + i][col + i] for i in range(4)]
            score += count_sequence(line, AI_PLAYER)
            score -= count_sequence(line, HUMAN_PLAYER)

    # Diagonal (negative slope)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            line = [board[row - i][col + i] for i in range(4)]
            score += count_sequence(line, AI_PLAYER)
            score -= count_sequence(line, HUMAN_PLAYER)

    return score

def max_value(board, alpha, beta, depth):
    """Maximize utility for AI."""
    if Terminal_Test(board):
        return Utility(board)
    if depth == 0:
        return heuristic(board)

    v = float('-inf')
    for col in prioritize_moves(board):  # Use prioritized move ordering
        new_board, _ = drop_piece(board, col, AI_PLAYER)
        v = max(v, min_value(new_board, alpha, beta, depth - 1))
        alpha = max(alpha, v)
        if v >= beta:
            return v
    return v

def min_value(board, alpha, beta, depth):
    """Minimize utility for human."""
    if Terminal_Test(board):
        return Utility(board)
    if depth == 0:
        return heuristic(board)

    v = float('inf')
    for col in prioritize_moves(board):  # Use prioritized move ordering
        new_board, _ = drop_piece(board, col, HUMAN_PLAYER)
        v = min(v, max_value(new_board, alpha, beta, depth - 1))
        beta = min(beta, v)
        if v <= alpha:
            return v
    return v

def IA_Decision(board):
    """Decide the best column for the AI."""
    start_time = time.time()
    best_value = float('-inf')
    best_move = None
    moves = prioritize_moves(board)  # Prioritize winning/blocking moves

    for col in moves:
        new_board, _ = drop_piece(board, col, AI_PLAYER)
        value = min_value(new_board, float('-inf'), float('inf'), MAX_DEPTH - 1)
        if value > best_value:
            best_value = value
            best_move = col
        if time.time() - start_time > 9:
            break

    return best_move if best_move is not None else random.choice(get_valid_moves(board))

def print_board(board):
    """Print the game board in a readable format."""
    print("\n  0  1  2  3  4  5  6  7  8  9 10 11")
    for row in range(ROWS):
        row_str = "|"
        for col in range(COLS):
            if board[row][col] == AI_PLAYER:
                row_str += " O "
            elif board[row][col] == HUMAN_PLAYER:
                row_str += " X "
            else:
                row_str += " . "
        row_str += "|"
        print(row_str)
    print("-" * (COLS * 3 + 3))

def play_game():
    """Main game loop for human vs AI."""
    board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
    
    while True:
        choice = input("Who starts? (1 for Human, 2 for AI): ").strip()
        if choice in ['1', '2']:
            human_starts = (choice == '1')
            break
        print("Invalid choice. Enter 1 for Human or 2 for AI.")

    current_player = HUMAN_PLAYER if human_starts else AI_PLAYER

    while not Terminal_Test(board):
        print_board(board)
        if current_player == HUMAN_PLAYER:
            while True:
                try:
                    col = int(input("Your move (column 0-11): ").strip())
                    if is_valid_move(board, col):
                        board, _ = drop_piece(board, col, HUMAN_PLAYER)
                        print(f"Human played in column {col}")
                        break
                    else:
                        print("Invalid move. Column is full or out of range.")
                except ValueError:
                    print("Please enter a number between 0 and 11.")
        else:
            col = IA_Decision(board)
            board, _ = drop_piece(board, col, AI_PLAYER)
            print(f"AI played in column {col}")

        if check_win(board, HUMAN_PLAYER):
            print_board(board)
            print("Human wins!")
            return
        elif check_win(board, AI_PLAYER):
            print_board(board)
            print("AI wins!")
            return
        elif is_board_full(board):
            print_board(board)
            print("It's a draw!")
            return

        current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

def test_blocking_threat():
    """Test if AI blocks a human's three-in-a-row threat."""
    board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
    # Set up human's three-in-a-row at the bottom (e.g., columns 0, 1, 2)
    board[5][0] = HUMAN_PLAYER
    board[5][1] = HUMAN_PLAYER
    board[5][2] = HUMAN_PLAYER
    # Column 3 is empty, so AI should play there to block
    print("Test board (human has three in a row at bottom 0-2):")
    print_board(board)
    col = IA_Decision(board)
    print(f"AI chose column {col}")
    assert col == 3, f"AI failed to block human's win (chose {col} instead of 3)"
    print("AI successfully blocked human's three-in-a-row!")

if __name__ == "__main__":
    print("Welcome to Connect Four (6x12 grid)!")
    # Uncomment to run the test
    # test_blocking_threat()
    play_game()