import streamlit as st
import numpy as np
import random

BOARD_SIZE = 10
SEQ = 5
HUMAN = 0
AI = 1


# -----------------------------
# Board utilities
# -----------------------------
def create_board():
    board = np.full((BOARD_SIZE, BOARD_SIZE), -1)
    empty = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)]
    return board, empty


def place(board, empty, move, player):
    x, y = move
    board[x, y] = player
    empty.remove(move)


def check_sequence(board, player, move):
    x, y = move
    dirs = [(1,0),(0,1),(1,1),(1,-1)]

    for dx,dy in dirs:
        count = 1

        for step in range(1,SEQ):
            nx,ny = x+dx*step,y+dy*step
            if 0<=nx<BOARD_SIZE and 0<=ny<BOARD_SIZE and board[nx,ny]==player:
                count+=1
            else:
                break

        for step in range(1,SEQ):
            nx,ny = x-dx*step,y-dy*step
            if 0<=nx<BOARD_SIZE and 0<=ny<BOARD_SIZE and board[nx,ny]==player:
                count+=1
            else:
                break

        if count>=SEQ:
            return True

    return False


# -----------------------------
# Simulation
# -----------------------------
def simulate_game():

    board, empty = create_board()
    player = HUMAN

    moves = []

    for _ in range(100):

        move = random.choice(empty)
        moves.append(move)

        place(board, empty, move, player)

        if check_sequence(board, player, move):
            return player, moves

        player = 1-player

    return -1, moves


# -----------------------------
# Markov Chain Construction
# -----------------------------
def build_markov_matrix(move_lists):

    size = BOARD_SIZE * BOARD_SIZE
    M = np.zeros((size, size))

    for moves in move_lists:
        for i in range(len(moves)-1):

            a = moves[i][0]*BOARD_SIZE + moves[i][1]
            b = moves[i+1][0]*BOARD_SIZE + moves[i+1][1]

            M[a][b] += 1

    row_sums = M.sum(axis=1)

    for i in range(len(row_sums)):
        if row_sums[i] > 0:
            M[i] /= row_sums[i]

    return M


# -----------------------------
# Run simulations
# -----------------------------
def run_simulations(num_games):

    results = []
    move_lists = []

    for _ in range(num_games):

        winner, moves = simulate_game()
        results.append(winner)
        move_lists.append(moves)

    results = np.array(results)

    ai_wins = np.sum(results == AI)
    human_wins = np.sum(results == HUMAN)

    winrate = ai_wins / num_games

    markov = build_markov_matrix(move_lists)

    eigvals, eigvecs = np.linalg.eig(markov)

    principal_vector = np.real(eigvecs[:,0])

    heatmap = principal_vector.reshape(BOARD_SIZE, BOARD_SIZE)

    return ai_wins, human_wins, winrate, heatmap


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Sequence AI Mathematical Analysis")

st.write("Fast Monte Carlo simulation with Markov chains and eigenvector analysis.")

num_games = st.slider(
    "Number of games to simulate",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

if st.button("Run Simulation"):

    ai_wins, human_wins, winrate, heatmap = run_simulations(num_games)

    st.subheader("Game Results")

    st.write("AI Wins:", ai_wins)
    st.write("Human Wins:", human_wins)
    st.write("AI Win Rate:", round(winrate,3))

    st.subheader("Eigenvector Board Influence Map")

    st.write(
        "This matrix shows which board positions dominate the Markov transition dynamics."
    )

    st.dataframe(heatmap)
