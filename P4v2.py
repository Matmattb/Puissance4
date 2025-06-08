import random
import time
import copy

# Constants
ROWS = 6
COLS = 12
AI_PLAYER = 1  # IA (rouge)
HUMAN_PLAYER = -1  # Humain ou adversaire IA (jaune)
EMPTY = 0
MAX_TIME = 9  # 9-second limit (leaving 1 sec for overhead)
BASE_DEPTH = 3  # Reduced base depth for faster response
DEBUG = False  # Activer le débogage pour analyse
MAX_PIONS = 42  # 42 pions au total

# Fonctions de base
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
    # Vérification horizontale
    for row in range(ROWS):
        for col in range(COLS - 3):
            if all(board[row][col + i] == player for i in range(4)):
                return True
    # Vérification verticale
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
    return all(board[0][col] != EMPTY for col in range(COLS))

def Terminal_Test(board):
    return check_win(board, AI_PLAYER) or check_win(board, HUMAN_PLAYER) or is_board_full(board)

# Fonctions d'évaluation et heuristiques
def evaluate_position(board):
    score = 0
    # Bonus pour les configurations de l'IA
    score += evaluate_lines(board, AI_PLAYER)
    # Pénalités pour les configurations de l'humain
    score -= evaluate_lines(board, HUMAN_PLAYER)
    return score

def evaluate_lines(board, player):
    score = 0
    # Évaluation horizontale
    for row in range(ROWS):
        for col in range(COLS - 3):
            line = [board[row][col + i] for i in range(4)]
            score += evaluate_line(line, player)
    # Évaluation verticale
    for col in range(COLS):
        for row in range(ROWS - 3):
            line = [board[row + i][col] for i in range(4)]
            score += evaluate_line(line, player)
    # Évaluation diagonale montante
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            line = [board[row + i][col + i] for i in range(4)]
            score += evaluate_line(line, player)
    # Évaluation diagonale descendante
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            line = [board[row - i][col + i] for i in range(4)]
            score += evaluate_line(line, player)
    return score

def evaluate_line(line, player):
    opponent = -player
    if line.count(opponent) > 0 and line.count(player) > 0:
        return 0  # ligne bloquée

    if line.count(player) == 4:
        return 1000  # victoire immédiate
    elif line.count(player) == 3 and line.count(EMPTY) == 1:
        return 50  # menace forte
    elif line.count(player) == 2 and line.count(EMPTY) == 2:
        return 10  # potentiel
    elif line.count(player) == 1 and line.count(EMPTY) == 3:
        return 1  # faible potentiel
    else:
        return 0

# Algorithme Minimax avec élagage Alpha-Beta
def minimax_ab(board, depth, alpha, beta, maximizing_player, start_time):
    if time.time() - start_time > MAX_TIME or depth == 0 or Terminal_Test(board):
        return evaluate_position(board)

    valid_moves = get_valid_moves(board)

    if maximizing_player:
        value = float('-inf')
        for col in valid_moves:
            new_board, _ = drop_piece(board, col, AI_PLAYER)
            value = max(value, minimax_ab(new_board, depth - 1, alpha, beta, False, start_time))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # élagage beta
        return value
    else:
        value = float('inf')
        for col in valid_moves:
            new_board, _ = drop_piece(board, col, HUMAN_PLAYER)
            value = min(value, minimax_ab(new_board, depth - 1, alpha, beta, True, start_time))
            beta = min(beta, value)
            if alpha >= beta:
                break  # élagage alpha
        return value

def IA_Decision(board):
    start_time = time.time()
    valid_moves = get_valid_moves(board)
    best_move = random.choice(valid_moves)  # initialisation aléatoire
    best_score = float('-inf')

    # Itérer sur les coups possibles et évaluer
    for col in valid_moves:
        new_board, _ = drop_piece(board, col, AI_PLAYER)
        score = minimax_ab(new_board, BASE_DEPTH, float('-inf'), float('inf'), False, start_time)
        
        if score > best_score:
            best_score = score
            best_move = col
        
        # Vérifier la contrainte de temps après chaque coup évalué
        if time.time() - start_time > MAX_TIME:
            print("Temps limite dépassé pendant la décision de l'IA")
            break
    
    # En cas de temps restant, augmenter la profondeur (optionnel)
    
    return best_move

# Fonctions d'affichage et de jeu
def print_board(board):
    print("\n  " + "  ".join(str(i) for i in range(COLS)))
    for row in board:
        print("| " + "  ".join('X' if cell == AI_PLAYER else 'O' if cell == HUMAN_PLAYER else '.' for cell in row) + " |")
    print("-" * (COLS*3 + 1))

def play_game():
    board = [[EMPTY]*COLS for _ in range(ROWS)]
    current_player = HUMAN_PLAYER if int(input("Qui commence? (1 pour Humain, 2 pour IA): ")) == 1 else AI_PLAYER
    
    while not Terminal_Test(board):
        print_board(board)
        
        if current_player == HUMAN_PLAYER:
            col = int(input(f"Votre coup (colonne 0-{COLS-1}): "))
            if col not in get_valid_moves(board):
                print("Coup invalide!")
                continue
        else:
            print("L'IA réfléchit...")
            col = IA_Decision(board)
            print(f"L'IA a joué dans la colonne {col}")
        
        board, _ = drop_piece(board, col, current_player)
        current_player = AI_PLAYER if current_player == HUMAN_PLAYER else HUMAN_PLAYER
    
    print_board(board)
    if check_win(board, AI_PLAYER):
        print("L'IA gagne!")
    elif check_win(board, HUMAN_PLAYER):
        print("L'Humain gagne!")
    else:
        print("Match nul!")

# Pour démarrer le jeu
if __name__ == "__main__":
    play_game()
