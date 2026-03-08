import numpy as np
import random

BOARD_SIZE = 10
SEQ = 5
NUM_GAMES = 20

HUMAN = 0
AI = 1


def create_board():
    board = np.full((BOARD_SIZE, BOARD_SIZE), -1)
    empty = {(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)}
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


def ai_move(empty):
    return random.choice(tuple(empty))


def simulate_game():
    board, empty = create_board()
    player = HUMAN

    for _ in range(100):

        move = ai_move(empty)
        place(board, empty, move, player)

        if check_sequence(board, player, move):
            return player

        player = 1-player

    return -1


def run_simulations():
    results = []

    for _ in range(NUM_GAMES):
        results.append(simulate_game())

    results = np.array(results)

    ai_wins = np.sum(results==AI)
    human_wins = np.sum(results==HUMAN)

    mean = np.mean(results==AI)
    var = np.var(results==AI)

    return {
        "AI_wins": ai_wins,
        "Human_wins": human_wins,
        "Mean_AI_winrate": mean,
        "Variance": var
    }


if __name__ == "__main__":

    stats = run_simulations()

    print("Results from",NUM_GAMES,"games")
    print(stats)
