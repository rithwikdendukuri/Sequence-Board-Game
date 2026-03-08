import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
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
        board[x, y] = 0.5
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
            # Increment heatmap only if player 1 is placing (can track for either player)
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

def run_simulations(num_games=1000):
    heatmap = np.zeros((BOARD_SIZE, BOARD_SIZE))
    results = []
    for _ in range(num_games):
        game = simulate_game("greedy", "probabilistic", heatmap)
        results.append(game)
    # Normalize heatmap
    heatmap = heatmap / np.max(heatmap)
    return results, heatmap

# -------------------------------
# Quant Analysis
# -------------------------------
def compute_statistics(results):
    num_moves = np.array([g["num_moves"] for g in results])
    winners = np.array([g["winner"] for g in results])

    mean_moves = np.mean(num_moves)
    std_moves = np.std(num_moves, ddof=1)
    ci_95 = stats.t.interval(0.95, len(num_moves)-1, loc=mean_moves, scale=std_moves/np.sqrt(len(num_moves)))

    win_rate_1 = np.mean(winners==1)
    win_rate_2 = np.mean(winners==2)
    z_stat, p_val = stats.proportions_ztest([np.sum(winners==1), np.sum(winners==2)], [len(winners), len(winners)])

    board_vectors = [g["final_board"].flatten() for g in results]
    board_matrix = np.array(board_vectors)
    pca = PCA(n_components=2)
    pca.fit(board_matrix)
    explained_variance = pca.explained_variance_ratio_

    table = np.zeros((2,2))
    for g in results:
        starting_player = 1
        winner = g["winner"]
        table[starting_player-1, winner-1] += 1
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(table)

    summary = {
        "mean_moves": mean_moves,
        "std_moves": std_moves,
        "ci_95_moves": ci_95,
        "win_rate_1": win_rate_1,
        "win_rate_2": win_rate_2,
        "z_test_win_rate": (z_stat, p_val),
        "pca_explained_variance": explained_variance,
        "chi2_starting_vs_winner": (chi2_stat, chi2_p)
    }
    return summary

# -------------------------------
# Heatmap Visualization
# -------------------------------
def plot_heatmap(heatmap):
    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Board Position Value Heatmap (Normalized)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()

# -------------------------------
# Run Pipeline
# -------------------------------
if __name__ == "__main__":
    print("Running 1000 simulations...")
    results, heatmap = run_simulations(num_games=1000)
    summary = compute_statistics(results)

    print("\n--- Quantitative Summary ---")
    print(f"Mean moves to win: {summary['mean_moves']:.2f} ± {summary['std_moves']:.2f}")
    print(f"95% CI for moves: {summary['ci_95_moves']}")
    print(f"Win rate Player 1: {summary['win_rate_1']:.2%}")
    print(f"Win rate Player 2: {summary['win_rate_2']:.2%}")
    print(f"Z-test for win rate difference: z={summary['z_test_win_rate'][0]:.3f}, p={summary['z_test_win_rate'][1]:.3f}")
    print(f"PCA explained variance (top 2 components): {summary['pca_explained_variance']}")
    print(f"Chi-square starting player vs winner: chi2={summary['chi2_starting_vs_winner'][0]:.3f}, p={summary['chi2_starting_vs_winner'][1]:.3f}")

    print("\nGenerating strategy heatmap...")
    plot_heatmap(heatmap)
