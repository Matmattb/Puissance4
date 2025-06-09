import random
import time
import copy

ROWS = 6
COLS = 12
IA = 1
ADVERSAIRE = -1
EMPTY = 0
MAX_TIME = 9  # 9 secondes pour éviter de dépasser la limite
BASE_DEPTH = 3
MAX_PIONS = 42

def is_valid_move(board, col):
    return 0 <= col < COLS and board[0][col] == EMPTY

def get_valid_moves(board):
    return [col for col in range(COLS) if is_valid_move(board, col)]

def drop_piece(board, col, player):
    new_board = copy.deepcopy(board)
    for row in range(ROWS - 1, -1, -1):
        if new_board[row][col] == EMPTY:
            new_board[row][col] = player
            return new_board, row
    return new_board, -1

def check_win(board, player):
    # Horizontale
    for row in range(ROWS):
        for col in range(COLS - 3):
            if all(board[row][col + i] == player for i in range(4)):
                return True
    # Verticale
    for col in range(COLS):
        for row in range(ROWS - 3):
            if all(board[row + i][col] == player for i in range(4)):
                return True
    # Diagonale montante
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if all(board[row + i][col + i] == player for i in range(4)):
                return True
    # Diagonale descendante
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            if all(board[row - i][col + i] == player for i in range(4)):
                return True
    return False

def is_board_full(board):
    count = 0
    for col in range(COLS):
        for row in range(ROWS):
            if board[row][col] != EMPTY:
                count += 1
            if count == MAX_PIONS:
                return True
    return False

def Terminal_Test(board):
    return check_win(board, IA) or check_win(board, ADVERSAIRE) or is_board_full(board)

def evaluate_line(line, player):
    opponent = -player
    if line.count(opponent) > 0 and line.count(player) > 0:
        return 0  # ligne bloquée

    if line.count(player) == 4:
        return 100000  # victoire immédiate
    elif line.count(player) == 3 and line.count(EMPTY) == 1:
        return 100  # menace forte
    elif line.count(player) == 2 and line.count(EMPTY) == 2:
        return 10  # potentiel
    elif line.count(player) == 1 and line.count(EMPTY) == 3:
        return 1  # faible potentiel
    else:
        return 0

def evaluate_lines(board, player):
    score = 0
    # Horizontale
    for row in range(ROWS):
        for col in range(COLS - 3):
            line = [board[row][col + i] for i in range(4)]
            score += evaluate_line(line, player)
    # Verticale
    for col in range(COLS):
        for row in range(ROWS - 3):
            line = [board[row + i][col] for i in range(4)]
            score += evaluate_line(line, player)
    # Diagonale montante
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            line = [board[row + i][col + i] for i in range(4)]
            score += evaluate_line(line, player)
    # Diagonale descendante
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            line = [board[row - i][col + i] for i in range(4)]
            score += evaluate_line(line, player)
    return score

def evaluate_position(board):
    # Offensive : IA, Défensive : ADVERSAIRE
    score = 0
    score += evaluate_lines(board, IA)
    score -= 1.2 * evaluate_lines(board, ADVERSAIRE)  # pondération : défend mais attaque d'abord
    # Bonus pour le centre
    center_col = COLS // 2
    for row in range(ROWS):
        if board[row][center_col] == IA:
            score += 3
    return score

def minimax_ab(board, depth, alpha, beta, maximizing_player, start_time):
    if time.time() - start_time > MAX_TIME or depth == 0 or Terminal_Test(board):
        return evaluate_position(board)
    valid_moves = get_valid_moves(board)
    if maximizing_player:
        value = float('-inf')
        for col in valid_moves:
            new_board, _ = drop_piece(board, col, IA)
            # Coup gagnant immédiat
            if check_win(new_board, IA):
                return 100000
            value = max(value, minimax_ab(new_board, depth - 1, alpha, beta, False, start_time))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for col in valid_moves:
            new_board, _ = drop_piece(board, col, ADVERSAIRE)
            # Coup gagnant immédiat pour l'adversaire
            if check_win(new_board, ADVERSAIRE):
                return -100000
            value = min(value, minimax_ab(new_board, depth - 1, alpha, beta, True, start_time))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

def IA_Decision(board):
    start_time = time.time()
    valid_moves = get_valid_moves(board)
    best_move = random.choice(valid_moves)
    best_score = float('-inf')

    # Priorité : coup gagnant immédiat
    for col in valid_moves:
        temp_board, _ = drop_piece(board, col, IA)
        if check_win(temp_board, IA):
            return col  # Joue pour gagner tout de suite
    # Priorité : bloquer l'adversaire
    for col in valid_moves:
        temp_board, _ = drop_piece(board, col, ADVERSAIRE)
        if check_win(temp_board, ADVERSAIRE):
            return col  # Bloque la victoire adverse

    # Sinon, évalue les coups
    for col in valid_moves:
        new_board, _ = drop_piece(board, col, IA)
        score = minimax_ab(new_board, BASE_DEPTH, float('-inf'), float('inf'), False, start_time)
        if score > best_score:
            best_score = score
            best_move = col
        if time.time() - start_time > MAX_TIME:
            break
    return best_move

def print_board(board):
    print("\n  " + "  ".join(str(i) for i in range(COLS)))
    for row in board:
        print("| " + "  ".join('X' if cell == IA else 'O' if cell == ADVERSAIRE else '.' for cell in row) + " |")
    print("-" * (COLS*3 + 1))

def play_game():
    board = [[EMPTY]*COLS for _ in range(ROWS)]
    current_player = ADVERSAIRE if int(input("Qui commence? (1 pour Humain, 2 pour IA): ")) == 1 else IA

    while not Terminal_Test(board):
        print_board(board)
        if current_player == ADVERSAIRE:
            col = int(input(f"Votre coup (colonne 0-{COLS-1}): "))
            if col not in get_valid_moves(board):
                print("Coup invalide!")
                continue
        else:
            print("L'IA réfléchit...")
            col = IA_Decision(board)
            print(f"L'IA a joué dans la colonne {col}")
        board, _ = drop_piece(board, col, current_player)
        current_player = IA if current_player == ADVERSAIRE else ADVERSAIRE

    print_board(board)
    if check_win(board, IA):
        print("L'IA gagne!")
    elif check_win(board, ADVERSAIRE):
        print("L'Humain gagne!")
    else:
        print("Match nul!")

if __name__ == "__main__":
    play_game()
