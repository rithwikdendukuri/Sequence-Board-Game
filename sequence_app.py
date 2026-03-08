import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

BOARD_SIZE = 10
SEQ = 5
HUMAN = 0
AI = 1


# -----------------------------
# Board Utilities
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
# Markov Matrix
# -----------------------------
def build_markov_matrix(move_lists):

    size = BOARD_SIZE * BOARD_SIZE
    M = np.zeros((size, size))

    for moves in move_lists:

        for i in range(len(moves)-1):

            a = moves[i][0]*BOARD_SIZE + moves[i][1]
            b = moves[i+1][0]*BOARD_SIZE + moves[i+1][1]

            M[a,b] += 1

    row_sums = M.sum(axis=1)

    for i in range(size):
        if row_sums[i] > 0:
            M[i] /= row_sums[i]

    return M


# -----------------------------
# Stationary Distribution
# -----------------------------
def stationary_distribution(P):

    eigvals, eigvecs = np.linalg.eig(P.T)

    idx = np.argmin(np.abs(eigvals - 1))

    vec = np.real(eigvecs[:, idx])
    vec = vec / vec.sum()

    return vec, eigvals


# -----------------------------
# Run simulations
# -----------------------------
def run_simulations(num_games):

    results = []
    move_lists = []
    move_counts = np.zeros((BOARD_SIZE,BOARD_SIZE))

    for _ in range(num_games):

        winner, moves = simulate_game()

        results.append(winner)
        move_lists.append(moves)

        for m in moves:
            move_counts[m] += 1

    results = np.array(results)

    ai_wins = np.sum(results == AI)
    human_wins = np.sum(results == HUMAN)

    winrate = ai_wins / num_games

    markov = build_markov_matrix(move_lists)

    stationary, eigvals = stationary_distribution(markov)

    influence_map = stationary.reshape(BOARD_SIZE,BOARD_SIZE)

    return ai_wins, human_wins, winrate, influence_map, move_counts, eigvals


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Sequence Strategy Mathematical Analysis")

st.write(
"Monte Carlo simulation with Markov chains and eigenvector analysis of board influence."
)

num_games = st.slider(
"Number of games",
10,
500,
100,
10
)

if st.button("Run Simulation"):

    ai_wins, human_wins, winrate, influence_map, move_counts, eigvals = run_simulations(num_games)

    st.subheader("Game Outcomes")

    fig1, ax1 = plt.subplots()
    ax1.bar(["AI Wins","Human Wins"], [ai_wins, human_wins])
    st.pyplot(fig1)

    st.write("AI Win Rate:", round(winrate,3))

    st.subheader("Move Frequency Heatmap")

    fig2, ax2 = plt.subplots()
    im = ax2.imshow(move_counts)
    plt.colorbar(im)
    st.pyplot(fig2)

    st.subheader("Markov Stationary Distribution (Board Influence)")

    fig3, ax3 = plt.subplots()
    im = ax3.imshow(influence_map)
    plt.colorbar(im)
    st.pyplot(fig3)

    st.subheader("Eigenvalue Spectrum")

    fig4, ax4 = plt.subplots()
    ax4.scatter(np.real(eigvals), np.imag(eigvals))
    ax4.set_xlabel("Real")
    ax4.set_ylabel("Imaginary")
    st.pyplot(fig4)
