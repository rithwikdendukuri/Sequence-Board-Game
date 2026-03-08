import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

BOARD = 10
SEQ = 5


# -----------------------------
# Board utilities
# -----------------------------

def create_board():
    board = np.full((BOARD,BOARD), -1)
    empty = [(i,j) for i in range(BOARD) for j in range(BOARD)]
    return board, empty


def place(board, empty, move, player):
    x,y = move
    board[x,y] = player
    empty.remove(move)


def check_sequence(board, player, move):

    x,y = move
    dirs = [(1,0),(0,1),(1,1),(1,-1)]

    for dx,dy in dirs:

        count = 1

        for step in range(1,SEQ):
            nx,ny = x+dx*step,y+dy*step
            if 0<=nx<BOARD and 0<=ny<BOARD and board[nx,ny]==player:
                count+=1
            else:
                break

        for step in range(1,SEQ):
            nx,ny = x-dx*step,y-dy*step
            if 0<=nx<BOARD and 0<=ny<BOARD and board[nx,ny]==player:
                count+=1
            else:
                break

        if count>=SEQ:
            return True

    return False


# -----------------------------
# Scoring functions
# -----------------------------

def adjacency(board,player,pos):

    x,y=pos
    score=0

    for dx in [-1,0,1]:
        for dy in [-1,0,1]:

            nx,ny=x+dx,y+dy

            if 0<=nx<BOARD and 0<=ny<BOARD:
                if board[nx,ny]==player:
                    score+=1

    return score


def line_extension(board,player,pos):

    x,y=pos
    dirs=[(1,0),(0,1),(1,1),(1,-1)]
    score=0

    for dx,dy in dirs:

        count=0

        for step in range(1,SEQ):
            nx,ny=x+dx*step,y+dy*step
            if 0<=nx<BOARD and 0<=ny<BOARD and board[nx,ny]==player:
                count+=1
            else:
                break

        for step in range(1,SEQ):
            nx,ny=x-dx*step,y-dy*step
            if 0<=nx<BOARD and 0<=ny<BOARD and board[nx,ny]==player:
                count+=1
            else:
                break

        score+=count

    return score


def heuristic_score(board,player,opponent,pos):

    extend=line_extension(board,player,pos)
    block=line_extension(board,opponent,pos)
    cluster=adjacency(board,player,pos)

    return 3*extend+2*block+cluster


# -----------------------------
# AI Strategies
# -----------------------------

def random_ai(board,empty,player):
    return random.choice(empty)


def greedy_ai(board,empty,player):

    scores=[adjacency(board,player,p) for p in empty]

    m=max(scores)

    candidates=[empty[i] for i,s in enumerate(scores) if s==m]

    return random.choice(candidates)


def heuristic_ai(board,empty,player):

    opponent=1-player

    scores=[heuristic_score(board,player,opponent,p) for p in empty]

    m=max(scores)

    candidates=[empty[i] for i,s in enumerate(scores) if s==m]

    return random.choice(candidates)


def probabilistic_ai(board,empty,player):

    opponent=1-player

    scores=np.array([heuristic_score(board,player,opponent,p) for p in empty])

    probs=np.exp(scores)/np.sum(np.exp(scores))

    idx=np.random.choice(len(empty),p=probs)

    return empty[idx]


STRATEGIES={
    "Random":random_ai,
    "Greedy":greedy_ai,
    "Heuristic":heuristic_ai,
    "Probabilistic":probabilistic_ai
}


# -----------------------------
# Simulation
# -----------------------------

def simulate_game(A,B):

    board,empty=create_board()

    player=0
    moves=[]

    for _ in range(100):

        if player==0:
            move=A(board,empty,player)
        else:
            move=B(board,empty,player)

        moves.append(move)

        place(board,empty,move,player)

        if check_sequence(board,player,move):
            return player,moves

        player=1-player

    return -1,moves


# -----------------------------
# Markov / Centrality
# -----------------------------

def build_markov(move_lists):

    N=BOARD*BOARD
    M=np.zeros((N,N))

    for moves in move_lists:

        for i in range(len(moves)-1):

            a=moves[i][0]*BOARD+moves[i][1]
            b=moves[i+1][0]*BOARD+moves[i+1][1]

            M[a,b]+=1

    row=M.sum(axis=1)

    for i in range(N):
        if row[i]>0:
            M[i]/=row[i]

    return M


def centrality(P):

    eigvals,eigvecs=np.linalg.eig(P.T)

    idx=np.argmin(np.abs(eigvals-1))

    vec=np.real(eigvecs[:,idx])
    vec/=vec.sum()

    return vec.reshape(BOARD,BOARD),eigvals


# -----------------------------
# Run simulations
# -----------------------------

def run_games(n,A,B):

    results=[]
    movesets=[]
    move_counts=np.zeros((BOARD,BOARD))
    seq_counts=np.zeros((BOARD,BOARD))

    for _ in range(n):

        winner,moves=simulate_game(A,B)

        results.append(winner)
        movesets.append(moves)

        for m in moves:
            move_counts[m]+=1

        if winner!=-1:
            for m in moves[-SEQ:]:
                seq_counts[m]+=1

    P=build_markov(movesets)

    central,eigs=centrality(P)

    seq_prob=seq_counts/np.maximum(move_counts,1)

    return results,move_counts,central,seq_prob,eigs


# -----------------------------
# UI
# -----------------------------

st.title("Sequence Strategy Explorer")

nerd=st.toggle("Nerd Mode (show math)")

A_choice=st.selectbox("Player A Strategy",list(STRATEGIES.keys()))
B_choice=st.selectbox("Player B Strategy",list(STRATEGIES.keys()))

games=st.slider("Number of games",20,500,100)

if st.button("Run Simulation"):

    results,move_counts,central,seq_prob,eigs=run_games(
        games,
        STRATEGIES[A_choice],
        STRATEGIES[B_choice]
    )

    winsA=sum(r==0 for r in results)
    winsB=sum(r==1 for r in results)

    tab1,tab2,tab3,tab4,tab5=st.tabs([
        "Results",
        "Move Heatmap",
        "Board Influence",
        "Sequence Probability",
        "Centrality Ranking"
    ])

    with tab1:

        fig,ax=plt.subplots()
        ax.bar(["Player A","Player B"],[winsA,winsB])
        st.pyplot(fig)

        if nerd:
            st.write("Win rate for player A:",winsA/(winsA+winsB))
            st.write("Win rate for player B:",winsB/(winsA+winsB))

    with tab2:

        fig,ax=plt.subplots()
        im=ax.imshow(move_counts)
        plt.colorbar(im)
        st.pyplot(fig)

    with tab3:

        fig,ax=plt.subplots()
        im=ax.imshow(central)
        plt.colorbar(im)
        st.pyplot(fig)

        if nerd:
            st.write("Stationary distribution of the transition matrix.")

    with tab4:

        fig,ax=plt.subplots()
        im=ax.imshow(seq_prob)
        plt.colorbar(im)
        st.pyplot(fig)

    with tab5:

        fig,ax=plt.subplots()
        ax.scatter(np.real(eigs),np.imag(eigs))
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginary")
        st.pyplot(fig)

        if nerd:
            st.write("Eigenvalue spectrum of the Markov transition matrix.")
