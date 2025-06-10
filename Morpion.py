import copy

class TicTacToe:
    def __init__(self):
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.ai_player = 1
        self.human_player = -1

    def print_board(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print("\n  0 1 2")
        for i, row in enumerate(self.board):
            print(f"{i} " + " ".join([symbols[cell] for cell in row]))

    def actions(self, state):
        return [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0]

    def result(self, state, action, player):
        new_state = copy.deepcopy(state)
        new_state[action[0]][action[1]] = player
        return new_state

    def terminal_test(self, state):
        # Victoire sur lignes/colonnes
        for i in range(3):
            if abs(sum(state[i])) == 3:
                return True
            if abs(sum(state[j][i] for j in range(3))) == 3:
                return True
        # Diagonales
        if abs(state[0][0] + state[1][1] + state[2][2]) == 3:
            return True
        if abs(state[0][2] + state[1][1] + state[2][0]) == 3:
            return True
        # Nul
        return all(cell != 0 for row in state for cell in row)

    def utility(self, state, player):
        # Victoire IA
        for i in range(3):
            if sum(state[i]) == 3:
                return 1
            if sum(state[i]) == -3:
                return -1
            if sum(state[j][i] for j in range(3)) == 3:
                return 1
            if sum(state[j][i] for j in range(3)) == -3:
                return -1
        diag1 = state[0][0] + state[1][1] + state[2][2]
        diag2 = state[0][2] + state[1][1] + state[2][0]
        if diag1 == 3 or diag2 == 3:
            return 1
        if diag1 == -3 or diag2 == -3:
            return -1
        # Nul
        return 0

    def evaluate_heuristic(self, state):
        # Heuristique simple : +10 pour 2 IA et 1 vide, -10 pour 2 humain et 1 vide
        score = 0
        lines = []

        # Lignes et colonnes
        for i in range(3):
            lines.append(state[i])  # lignes
            lines.append([state[j][i] for j in range(3)])  # colonnes
        # Diagonales
        lines.append([state[0][0], state[1][1], state[2][2]])
        lines.append([state[0][2], state[1][1], state[2][0]])

        for line in lines:
            if line.count(self.ai_player) == 2 and line.count(0) == 1:
                score += 10
            if line.count(self.human_player) == 2 and line.count(0) == 1:
                score -= 10
            if line.count(self.ai_player) == 1 and line.count(0) == 2:
                score += 1
            if line.count(self.human_player) == 1 and line.count(0) == 2:
                score -= 1
        return score

    def minimax_ab(self, state, depth, alpha, beta, maximizing):
        if self.terminal_test(state) or depth == 0:
            util = self.utility(state, self.ai_player)
            if util != 0 or self.terminal_test(state):
                return util * 100  # Victoire/défaite/nul
            else:
                return self.evaluate_heuristic(state)
        if maximizing:
            max_eval = float('-inf')
            for action in self.actions(state):
                eval = self.minimax_ab(self.result(state, action, self.ai_player), depth-1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for action in self.actions(state):
                eval = self.minimax_ab(self.result(state, action, self.human_player), depth-1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def best_move(self):
        best_score = float('-inf')
        best_action = None
        for action in self.actions(self.board):
            score = self.minimax_ab(self.result(self.board, action, self.ai_player), 6, float('-inf'), float('inf'), False)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def play(self):
        current_player = self.human_player if int(input("Qui commence ? (1 pour Humain, 2 pour IA): ")) == 1 else self.ai_player
        while True:
            self.print_board()
            if current_player == self.human_player:
                try:
                    row = int(input("Ligne (0-2): "))
                    col = int(input("Colonne (0-2): "))
                except Exception:
                    print("Entrée invalide !")
                    continue
                if not (0 <= row <= 2 and 0 <= col <= 2) or self.board[row][col] != 0:
                    print("Coup invalide!")
                    continue
                self.board[row][col] = self.human_player
            else:
                print("L'IA réfléchit...")
                move = self.best_move()
                self.board[move[0]][move[1]] = self.ai_player
            if self.terminal_test(self.board):
                break
            current_player = self.ai_player if current_player == self.human_player else self.human_player
        self.print_board()
        result = self.utility(self.board, self.ai_player)
        if result == 1:
            print("L'IA a gagné!")
        elif result == -1:
            print("Vous avez gagné!")
        else:
            print("Match nul!")

# Pour démarrer le jeu
if __name__ == "__main__":
    game = TicTacToe()
    game.play()
