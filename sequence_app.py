import streamlit as st
import random
from copy import deepcopy
import numpy as np

# --------------------------
# Sequence Board Setup
# --------------------------
board_template = [
    ["WILD", "2H", "3H", "4H", "5H", "6H", "7H", "8H", "9H", "WILD"],
    ["10H","JH","QH","KH","AH","2C","3C","4C","5C","10C"],
    ["6H","7H","8H","9H","10H","JC","QC","KC","AC","6C"],
    ["7H","8H","9H","10H","JH","QC","KC","AC","2D","7C"],
    ["8H","9H","10H","JH","QH","KC","AC","2D","3D","8C"],
    ["9D","10D","JD","QD","KD","AD","2S","3S","4S","9S"],
    ["10D","JD","QD","KD","AD","2S","3S","4S","5S","10S"],
    ["JD","QD","KD","AD","2S","3S","4S","5S","6S","JS"],
    ["QD","KD","AD","2S","3S","4S","5S","6S","7S","QS"],
    ["WILD", "8S", "9S", "10S", "JS", "QS", "KS", "AS", "2H", "WILD"]
]

wild_positions = [(0,0),(0,9),(9,0),(9,9)]
rows, cols = 10, 10

# --------------------------
# Deck creation
# --------------------------
def create_deck():
    suits = ['H','C','D','S']
    values = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    deck = []
    for v in values:
        if v=="J":
            deck.extend([v+s for s in suits])
        else:
            deck.extend([v+s for s in suits]*2)
    random.shuffle(deck)
    return deck

# --------------------------
# Hand size
# --------------------------
def get_hand_size(num_players):
    if num_players in [3,4]:
        return 6
    elif num_players == 6:
        return 5
    elif num_players in [8,9]:
        return 4
    elif num_players in [10,12]:
        return 3
    else:
        raise ValueError("Unsupported number of players.")

# --------------------------
# Deal hands
# --------------------------
def deal_hands(deck, num_players):
    hand_size = get_hand_size(num_players)
    hands = {}
    for i in range(1,num_players+1):
        hands[i] = [deck.pop() for _ in range(hand_size)]
    return hands

# --------------------------
# Board rendering
# --------------------------
def render_board(board, highlights=[]):
    st.subheader("Board")
    for r,row in enumerate(board):
        cols_stream = st.columns(len(row))
        for c,cell in enumerate(row):
            label = cell
            if (r,c) in highlights:
                label = f"**{cell}**"
            cols_stream[c].markdown(label, unsafe_allow_html=True)

# --------------------------
# Render hand visually
# --------------------------
def render_hand(hand):
    cols_stream = st.columns(len(hand))
    for i,card in enumerate(hand):
        cols_stream[i].markdown(f"🂠 {card}", unsafe_allow_html=True)

# --------------------------
# Heuristic functions
# --------------------------
def count_potential_sequences(board, r, c, player_chip):
    # Check all 8 directions for potential sequences
    directions = [(1,0),(0,1),(1,1),(1,-1),(-1,0),(0,-1),(-1,-1),(-1,1)]
    score = 0
    for dr, dc in directions:
        line_score = 0
        for k in range(1,5):
            rr, cc = r+dr*k, c+dc*k
            if 0<=rr<rows and 0<=cc<cols:
                if board[rr][cc] in [player_chip,"WILD"]:
                    line_score += 1
                elif board[rr][cc].startswith("X"):
                    break
        score += line_score
    return score

def ai_heuristic_move(board, hand, player_num):
    best_score = -1
    best_card = None
    best_pos = None
    player_chip = f"X{player_num}"
    
    for card in hand:
        for r in range(rows):
            for c in range(cols):
                if board[r][c]==card or (card in ["JS","JC"] and board[r][c] not in wild_positions):
                    score = count_potential_sequences(board, r, c, player_chip)
                    # simple: prioritize higher score positions
                    if score > best_score:
                        best_score = score
                        best_card = card
                        best_pos = (r,c)
    return best_card, best_pos

# --------------------------
# Streamlit UI
# --------------------------
st.title("Sequence Simulator: AI vs AI with Heuristics")

num_players = st.selectbox("Number of players", options=[3,4,6,8,9,10,12])
deck = create_deck()
hands = deal_hands(deck, num_players)
board_state = deepcopy(board_template)

render_board(board_state)

# --------------------------
# Auto-play one round
# --------------------------
for player in range(1,num_players+1):
    st.subheader(f"Player {player}'s Turn")
    hand = hands[player]
    render_hand(hand)
    card, pos = ai_heuristic_move(board_state, hand, player)
    if card is None or pos is None:
        continue
    r,c = pos
    
    # Handle Jacks
    if card in ["JH","JD"]:  # one-eyed jacks remove
        for rr in range(rows):
            for cc in range(cols):
                if board_state[rr][cc].startswith("X") and board_state[rr][cc] != f"X{player}":
                    board_state[rr][cc] = board_template[rr][cc]
                    break
    else:
        board_state[r][c] = f"X{player}"
    
    hand.remove(card)
    render_board(board_state, highlights=[pos])
    st.write(f"Player {player} played {card} at {pos}")
