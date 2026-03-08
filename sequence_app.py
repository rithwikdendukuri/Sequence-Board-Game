import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Constants & Game Setup
# -------------------------------
BOARD_SIZE = 10
SEQUENCE_LENGTH = 5
CORNER_POS = [(0, 0), (0, 9), (9, 0), (9, 9)]

STRATEGIES = ["random", "greedy", "probabilistic"]

# -------------------------------
# Helper Functions
# -------------------------------
def initialize_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for (x, y) in CORNER_POS:
        board[x, y] = 0.5  # corners count as wild
    return board

def check_sequence(board, player_id):
    count = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE - SEQUENCE_LENGTH + 1):
            if np.all(board[i, j:j+SEQUENCE_LENGTH] == player_id) or np.sum(board[i, j:j+SEQUENCE_LENGTH] == 0.5) > 0:
                count += 1
            if np.all(board[j:j+SEQUENCE_LENGTH, i] == player_id) or np.sum(board[j:j+SEQUENCE_LENGTH, i] == 0.5) > 0:
                count += 1
    for i in range(BOARD_SIZE - SEQUENCE_LENGTH + 1):
        for j in range(BOARD_SIZE - SEQUENCE_LENGTH + 1):
            diag1 = [board[i+k, j+k] for k in range(SEQUENCE_LENGTH)]
            diag2 = [board[i+k, j+SEQUENCE_LENGTH-1-k] for k in range(SEQUENCE_LENGTH)]
            if np.all(np.array(diag1) == player_id) or np.sum(np.array(diag1) == 0.5) > 0:
                count += 1
            if np.all(np.array(diag2) == player_id) or np.sum(np.array(diag2) == 0.5) > 0:
                count += 1
    return count

def get_available_moves(board):
    return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i, j] == 0]

def select_move(board, player_id, strategy):
    moves = get_available_moves(board)
    if not moves:
        return None
    if strategy == "random":
        return moves[np.random.choice(len(moves))]
    elif strategy == "greedy":
        best_score = -1
        best_move = moves[0]
        for move in moves:
            temp = board.copy()
            temp[move] = player_id
            score = check_sequence(temp, player_id)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
    elif strategy == "probabilistic":
        scores = []
        for move in moves:
            x, y = move
            adjacent = board[max(0,x-1):min(BOARD_SIZE,x+2), max(0,y-1):min(BOARD_SIZE,y+2)]
            score = np.sum(adjacent == player_id)
            scores.append(score + 1e-3*np.random.rand())
        scores = np.array(scores)
        probs = scores / np.sum(scores)
        return moves[np.random.choice(len(moves), p=probs)]

# -------------------------------
# Simulation Engine with Heatmap Tracking
# -------------------------------
def simulate_game(strategy1, strategy2, heatmap=None):
    board = initialize_board()
    moves = []
    player_ids = [1, 2]
    strategies = [strategy1, strategy2]
    current_player = 0

    while True:
        player_id = player_ids[current_player]
        move = select_move(board, player_id, strategies[current_player])
        if move is None:
            break
        board[move] = player_id
        moves.append((current_player+1, move))
        if heatmap is not None:
            heatmap[move] += 1
        sequences = check_sequence(board, player_id)
        if sequences >= 2:
            winner = current_player+1
            break
        current_player = 1 - current_player

    game_stats = {
        "winner": winner,
        "num_moves": len(moves),
        "move_sequence": moves,
        "final_board": board
    }
    return game_stats

def run_simulations(num_games=500):
    heatmap_greedy = np.zeros((BOARD_SIZE, BOARD_SIZE))
    heatmap_prob = np.zeros((BOARD_SIZE, BOARD_SIZE))
    results_greedy = []
    results_prob = []

    for _ in range(num_games):
        game = simulate_game("greedy", "probabilistic", heatmap_greedy)
        results_greedy.append(game)
        game = simulate_game("probabilistic", "greedy", heatmap_prob)
        results_prob.append(game)

    heatmap_greedy /= np.max(heatmap_greedy)
    heatmap_prob /= np.max(heatmap_prob)

    return results_greedy, results_prob, heatmap_greedy, heatmap_prob

# -------------------------------
# Manual Statistics Functions
# -------------------------------
def mean_std_ci(data):
    n = len(data)
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    ci = (mean_val - 1.96*std_val/np.sqrt(n), mean_val + 1.96*std_val/np.sqrt(n))
    return mean_val, std_val, ci

def z_test_proportion(success_a, success_b, n):
    p1, p2 = success_a/n, success_b/n
    p_pool = (success_a+success_b)/(2*n)
    z = (p1 - p2)/np.sqrt(p_pool*(1-p_pool)*(2/n))
    return z

def chi2_test(table):
    total = np.sum(table)
    expected = np.outer(np.sum(table, axis=1), np.sum(table, axis=0)) / total
    chi2 = np.sum((table - expected)**2 / expected)
    return chi2

# -------------------------------
# Quant Analysis
# -------------------------------
def compute_statistics(results):
    num_moves = np.array([g["num_moves"] for g in results])
    winners = np.array([g["winner"] for g in results])

    mean_moves, std_moves, ci_moves = mean_std_ci(num_moves)
    win_rate_1 = np.mean(winners==1)
    win_rate_2 = np.mean(winners==2)
    z_stat = z_test_proportion(np.sum(winners==1), np.sum(winners==2), len(winners))

    table = np.zeros((2,2))
    for g in results:
        table[0, g["winner"]-1] += 1
    chi2_stat = chi2_test(table)

    summary = {
        "mean_moves": mean_moves,
        "std_moves": std_moves,
        "ci_moves": ci_moves,
        "win_rate_1": win_rate_1,
        "win_rate_2": win_rate_2,
        "z_test_win_rate": z_stat,
        "chi2_starting_vs_winner": chi2_stat
    }
    return summary

# -------------------------------
# Heatmap Visualization
# -------------------------------
def plot_heatmaps(heatmap1, heatmap2):
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    sns.heatmap(heatmap1, annot=False, cmap="Reds", ax=axes[0])
    axes[0].set_title("Greedy AI Board Value")
    sns.heatmap(heatmap2, annot=False, cmap="Blues", ax=axes[1])
    axes[1].set_title("Probabilistic AI Board Value")
    plt.show()

# -------------------------------
# Run Pipeline
# -------------------------------
if __name__ == "__main__":
    print("Running simulations for heatmap comparison...")
    results_greedy, results_prob, heat_greedy, heat_prob = run_simulations(num_games=500)

    summary_greedy = compute_statistics(results_greedy)
    summary_prob = compute_statistics(results_prob)

    print("\n--- Greedy AI Summary ---")
    print(summary_greedy)
    print("\n--- Probabilistic AI Summary ---")
    print(summary_prob)

    print("\nGenerating side-by-side strategy heatmaps...")
    plot_heatmaps(heat_greedy, heat_prob)
