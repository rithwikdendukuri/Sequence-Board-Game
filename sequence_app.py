import numpy as np
import random
from collections import Counter
from itertools import product

# -----------------------------
# Constants
# -----------------------------
BOARD_SIZE = 10
SEQUENCE_LENGTH = 5
NUM_SIMULATIONS = 50  # 50 games per AI strategy
HUMAN = 0
AI = 1

# Card representation simplified for quant analysis
SUITS = ['♠', '♥', '♦', '♣']
NUM_CARDS = list(range(2, 11)) + ['J', 'Q', 'K', 'A']

# -----------------------------
# Board & Game Utilities
# -----------------------------
def create_board():
    """Create a 10x10 board with None."""
    return np.full((BOARD_SIZE, BOARD_SIZE), None)

def place_chip(board, pos, player):
    """Place a chip if legal."""
    x, y = pos
    if board[x, y] is None:
        board[x, y] = player
        return True
    return False

def check_sequence(board, player, last_move):
    """Check if last move formed a sequence."""
    x, y = last_move
    directions = [(1,0),(0,1),(1,1),(1,-1)]
    for dx, dy in directions:
        count = 1
        # Forward
        for step in range(1, SEQUENCE_LENGTH):
            nx, ny = x+dx*step, y+dy*step
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx, ny]==player:
                count +=1
            else:
                break
        # Backward
        for step in range(1, SEQUENCE_LENGTH):
            nx, ny = x-dx*step, y-dy*step
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and board[nx, ny]==player:
                count +=1
            else:
                break
        if count >= SEQUENCE_LENGTH:
            return True
    return False

def legal_moves(board):
    """Return list of empty positions."""
    return [(i,j) for i,j in product(range(BOARD_SIZE), repeat=2) if board[i,j] is None]

# -----------------------------
# AI Utilities
# -----------------------------
def simple_ai(board, player):
    """AI chooses a move near existing chips (basic heuristic)."""
    moves = legal_moves(board)
    if not moves:
        return None
    # Favor positions adjacent to own chips
    weights = []
    for x,y in moves:
        score = 0
        for dx,dy in product([-1,0,1],repeat=2):
            nx, ny = x+dx, y+dy
            if 0<=nx<BOARD_SIZE and 0<=ny<BOARD_SIZE and board[nx,ny]==player:
                score +=1
        weights.append(score+1)  # avoid 0
    chosen = random.choices(moves, weights=weights, k=1)[0]
    print(f"AI ({player}) thinking... chooses move {chosen}")
    return chosen

# -----------------------------
# Simulation & Statistics
# -----------------------------
def simulate_game(human_first=True, human_strategy=False):
    """Simulate a full game, return winner and move matrix."""
    board = create_board()
    move_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE))
    player_turn = HUMAN if human_first else AI
    moves_played = 0
    winner = None
    
    while moves_played < BOARD_SIZE*BOARD_SIZE:
        if player_turn == HUMAN and human_strategy:
            # For research, simulate human randomly
            move = random.choice(legal_moves(board))
        else:
            move = simple_ai(board, player_turn)
        
        if move is None:
            break
        place_chip(board, move, player_turn)
        move_matrix[move] +=1
        moves_played +=1
        
        if check_sequence(board, player_turn, move):
            winner = player_turn
            break
        player_turn = AI if player_turn==HUMAN else HUMAN
    return winner, move_matrix

def run_simulations(num_games=NUM_SIMULATIONS, human_strategy=False):
    """Run multiple games and collect statistics."""
    results = []
    combined_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for _ in range(num_games):
        winner, move_matrix = simulate_game(human_strategy=human_strategy)
        results.append(winner)
        combined_matrix += move_matrix
    results = np.array(results)
    
    # Sample statistics
    mean_seq = np.mean(results==AI)
    var_seq = np.var(results==AI)
    
    # Covariance of moves
    flat_matrix = combined_matrix.flatten()
    cov = np.cov(flat_matrix, rowvar=False)
    
    # PCA (eigenvectors of covariance)
    eigvals, eigvecs = np.linalg.eig(cov.reshape(-1,1))
    
    return {
        "win_rate_AI": mean_seq,
        "variance": var_seq,
        "combined_move_matrix": combined_matrix,
        "covariance": cov,
        "pca_eigenvalues": eigvals
    }

# -----------------------------
# Main Execution
# -----------------------------
if __name__=="__main__":
    print("Running AI vs AI simulations for research analysis...")
    stats_ai = run_simulations()
    print("AI Win Rate:", stats_ai['win_rate_AI'])
    print("Variance of wins:", stats_ai['variance'])
    print("Top PCA Eigenvalue:", stats_ai['pca_eigenvalues'][0])
    print("Combined Move Matrix:\n", stats_ai['combined_move_matrix'])
    
    print("\nHuman vs AI simulation (random human moves for testing)...")
    stats_human = run_simulations(human_strategy=True)
    print("AI Win Rate:", stats_human['win_rate_AI'])
    print("Variance of wins:", stats_human['variance'])
    print("Top PCA Eigenvalue:", stats_human['pca_eigenvalues'][0])
