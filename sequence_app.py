"""
Sequence Simulator — hyper-realistic Streamlit app with full rules, heuristic AI, batch simulation,
and data logging for probability / ML research.

Requirements:
  pip install streamlit pillow numpy pandas

Save as `sequence_simulator.py` and run:
  streamlit run sequence_simulator.py

Notes:
- This is a single-file research prototype. It draws card images on-the-fly (Pillow) so no external
  image assets are required.
- The AI uses a weighted combination of 10 heuristics; weights are adjustable in the sidebar.
- Batch mode runs many games without rendering to collect statistics quickly.
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import random, time, io
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Set

# -------------------------
# Config / Constants
# -------------------------
ROWS, COLS = 10, 10
CARD_W, CARD_H = 72, 100           # single card image size in px (feel free to enlarge)
PADDING = 6
BOARD_BG = (24, 80, 40)            # tabletop green
PLAYER_COLORS = [(220, 40, 40), (40, 120, 220), (220, 180, 40), (120, 40, 180),
                 (30, 180, 100), (140, 90, 30), (60,60,60), (180, 60, 100)]
# One-eyed jacks (commonly J♥ and J♠); two-eyed jacks (J♦, J♣)
ONE_EYED = {"J♥", "J♠"}
TWO_EYED = {"J♦", "J♣"}
CORNER_POS = {(0,0),(0,9),(9,0),(9,9)}

# Standard deck representation
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]
STANDARD_DECK = [f"{r}{s}" for r in RANKS for s in SUITS]  # 52 unique ids

# -------------------------
# Utilities: Fonts and drawing
# -------------------------
# Try to get a TTF font for nicer rendering; fall back to default if unavailable.
def _get_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

FONT_SMALL = _get_font(12)
FONT_MED = _get_font(16)
FONT_LARGE = _get_font(22)

def draw_card_image(card: str,
                    chip_owner: Optional[int]=None,
                    highlight: bool=False,
                    corner_wild: bool=False) -> Image.Image:
    """
    Draw a single card image (rank+suit). If chip_owner present, draw a colored token overlay.
    If corner_wild True, draw a special wild-corner style.
    """
    img = Image.new("RGB", (CARD_W, CARD_H), color="white")
    draw = ImageDraw.Draw(img)
    # border
    border_color = (200,200,200) if not highlight else (255,215,50)
    draw.rectangle([1,1,CARD_W-2,CARD_H-2], outline=border_color, width=2)

    # card center: for WILD, special design
    if card == "WILD" or corner_wild:
        draw.rectangle([6,22,CARD_W-6,CARD_H-14], fill=(240,240,210))
        draw.text((CARD_W//2, CARD_H//2-8), "WILD", anchor="mm", font=FONT_MED, fill=(180,30,30))
        draw.ellipse([CARD_W//2-18, CARD_H-12, CARD_W//2+18, CARD_H-36], outline=(200,100,0), width=3)
    else:
        # draw rank and suit
        # determine suit glyph color
        rank = card[:-1]
        suit = card[-1]
        suit_symbol = suit  # using unicode suit characters in representation
        suit_color = (0,0,0) if suit in ["♠","♣"] else (180,0,0)
        draw.text((8,8), rank, font=FONT_SMALL, fill=suit_color)
        draw.text((CARD_W-10, CARD_H-18), suit_symbol, font=FONT_SMALL, fill=suit_color, anchor="rm")
        # center big suit
        draw.text((CARD_W//2, CARD_H//2-6), suit_symbol, anchor="mm", font=FONT_LARGE, fill=suit_color)
        # small rank bottom-left
        draw.text((8, CARD_H-18), rank, font=FONT_SMALL, fill=suit_color)

    # overlay chip
    if chip_owner is not None and chip_owner > 0:
        color = PLAYER_COLORS[(chip_owner-1) % len(PLAYER_COLORS)]
        # draw a filled circle at top-right
        cx, cy, r = CARD_W-18, 18, 12
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color, outline=(0,0,0))
        draw.text((cx, cy), str(chip_owner), anchor="mm", font=FONT_SMALL, fill=(255,255,255))
    return img

def compose_board_image(board_grid: List[List[str]],
                        chips: List[List[int]],
                        highlights: List[Tuple[int,int]]=[]) -> Image.Image:
    """
    Compose the full board as a single PIL image with cards and token overlays.
    """
    width = COLS*(CARD_W+PADDING) + PADDING
    height = ROWS*(CARD_H+PADDING) + PADDING
    bg = Image.new("RGB", (width, height), color=BOARD_BG)
    x0 = PADDING
    y0 = PADDING
    for r in range(ROWS):
        x = x0
        for c in range(COLS):
            card = board_grid[r][c]
            chip_owner = chips[r][c] if chips is not None else None
            highlight = (r,c) in highlights
            corner_wild = (r,c) in CORNER_POS
            card_img = draw_card_image(card, chip_owner=chip_owner, highlight=highlight, corner_wild=corner_wild)
            bg.paste(card_img, (x, y))
            x += CARD_W + PADDING
        y0 += CARD_H + PADDING
    return bg

# -------------------------
# Board generation: official mapping
# -------------------------
def make_official_board(seed: Optional[int]=None) -> List[List[str]]:
    """
    Create a 10x10 board where:
    - corners are "WILD"
    - every non-jack card (48 cards) is placed exactly twice in the 96 non-corner cells
    - jacks are not placed on the board
    """
    if seed is not None:
        random.seed(seed)
    non_jacks = [c for c in STANDARD_DECK if not c.startswith("J")]
    tokens = non_jacks * 2  # 96 tokens
    assert len(tokens) == 96
    random.shuffle(tokens)
    grid = [["" for _ in range(COLS)] for _ in range(ROWS)]
    # fill positions excluding corners row-major
    idx = 0
    for r in range(ROWS):
        for c in range(COLS):
            if (r,c) in CORNER_POS:
                grid[r][c] = "WILD"
            else:
                grid[r][c] = tokens[idx]
                idx += 1
    return grid

# -------------------------
# Deck & setup helpers
# -------------------------
def create_deck() -> List[str]:
    """Use two full decks (jacks included twice) — realistic supply for draws."""
    deck = STANDARD_DECK.copy() + STANDARD_DECK.copy()
    random.shuffle(deck)
    return deck

def get_hand_size(num_players: int) -> int:
    if num_players in [3,4]:
        return 6
    elif num_players == 6:
        return 5
    elif num_players in [8,9]:
        return 4
    elif num_players in [10,12]:
        return 3
    else:
        raise ValueError("unsupported player count")

# -------------------------
# Sequence detection utilities
# -------------------------
def team_of(player: int, player_to_team: Dict[int,int]) -> int:
    return player_to_team[player]

def find_sequence_positions(chips: List[List[int]], player_to_team: Dict[int,int], team_id: int, min_len: int=5) -> Set[Tuple[int,int]]:
    """
    Return set of positions that belong to sequences for given team.
    Corner WILD counts for every team automatically (we include corners in sequences).
    """
    rows_, cols_ = ROWS, COLS
    dirs = [(0,1),(1,0),(1,1),(-1,1)]
    matches = lambda rr,cc: ((rr,cc) in CORNER_POS) or (0 <= rr < rows_ and 0 <= cc < cols_ and chips[rr][cc] != 0 and player_to_team[chips[rr][cc]]==team_id)
    seqpos = set()
    for r in range(rows_):
        for c in range(cols_):
            if not matches(r,c): continue
            for dr,dc in dirs:
                pr, pc = r-dr, c-dc
                if 0<=pr<rows_ and 0<=pc<cols_ and matches(pr,pc):
                    continue  # not start
                run=[]
                rr, cc = r, c
                while 0<=rr<rows_ and 0<=cc<cols_ and matches(rr,cc):
                    run.append((rr,cc))
                    rr += dr; cc += dc
                if len(run) >= min_len:
                    seqpos.update(run)
    return seqpos

def count_team_sequences(chips: List[List[int]], player_to_team: Dict[int,int]) -> Dict[int,int]:
    teams = set(player_to_team.values())
    out = {}
    for t in teams:
        pos = find_sequence_positions(chips, player_to_team, t)
        out[t] = max(0, len(pos)//5)  # approximate count by positions/5 (distinct runs locked)
    return out

# -------------------------
# Legal actions & environment
# -------------------------
Action = Tuple[str, Optional[str], Tuple[int,int]]  # ("play_card", card, (r,c)) or ("turn_in_dead", card, (-1,-1))

def positions_of_card_on_board(board: List[List[str]], card: str) -> List[Tuple[int,int]]:
    out=[]
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == card:
                out.append((r,c))
    return out

def legal_actions(state: dict, player_id: int) -> List[Action]:
    """
    state contains: board, chips, hands, deck, completed_positions, player_to_team
    """
    board = state['board']; chips = state['chips']; hands = state['hands']; completed = state['completed']
    hand = hands[player_id]
    actions=[]
    for card in hand:
        if card.startswith("J"):
            # play a jack: one-eyed removes an opponent chip; two-eyed places anywhere open
            if card in TWO_EYED:
                for r in range(ROWS):
                    for c in range(COLS):
                        if chips[r][c]==0 and (r,c) not in completed and (r,c) not in CORNER_POS:
                            actions.append(("play_jack_two", card, (r,c)))
            else:
                # one-eyed: remove opponent chip not in completed and not corner
                for r in range(ROWS):
                    for c in range(COLS):
                        if (r,c) in CORNER_POS or (r,c) in completed: continue
                        owner = chips[r][c]
                        if owner != 0 and owner != player_id:
                            actions.append(("play_jack_one", card, (r,c)))
        else:
            poss = positions_of_card_on_board(board, card)
            placed=False
            for (r,c) in poss:
                if chips[r][c]==0 and (r,c) not in completed: 
                    actions.append(("play_card", card, (r,c)))
                    placed=True
            if not placed:
                # dead card turn-in
                actions.append(("turn_in_dead", card, (-1,-1)))
    return actions

def apply_action(state: dict, player_id: int, action: Action, draw_after=True):
    """
    Mutates a deep copy of state and returns new_state, reward, done, info
    reward: number of new sequences completed for player's team
    """
    s = deepcopy(state)
    typ, card, pos = action
    chips = s['chips']
    completed = s['completed']
    team_map = s['player_to_team']
    current_team = team_map[player_id]
    prev_team_seq = count_team_sequences(chips, team_map).get(current_team, 0)

    if typ == "play_card":
        r,c = pos
        # place chip
        s['hands'][player_id].remove(card)
        s['discard'].append(card)
        chips[r][c] = player_id
    elif typ == "play_jack_two":
        r,c = pos
        s['hands'][player_id].remove(card)
        s['discard'].append(card)
        chips[r][c] = player_id
    elif typ == "play_jack_one":
        r,c = pos
        # remove opponent chip
        s['hands'][player_id].remove(card)
        s['discard'].append(card)
        chips[r][c] = 0
    elif typ == "turn_in_dead":
        s['hands'][player_id].remove(card)
        s['discard'].append(card)
        # draw replacement below if deck non-empty
    else:
        raise ValueError("unknown action type")

    # draw
    if draw_after and s['deck']:
        s['hands'][player_id].append(s['deck'].pop(0))

    # recompute sequences and lock completed positions
    seqpos = find_sequence_positions(chips, team_map, current_team)
    if seqpos:
        s['completed'].update(seqpos)
    # update seq counts
    seq_counts = count_team_sequences(chips, team_map)
    s['seq_counts'] = seq_counts

    # done?
    num_teams = len(set(team_map.values()))
    required = 2 if num_teams==2 else 1
    done = any(v>=required for v in seq_counts.values())

    new_team_seq = seq_counts.get(current_team, 0)
    reward = max(0, new_team_seq - prev_team_seq)
    # advance player
    idx = s['players'].index(player_id)
    s['current_player'] = s['players'][(idx+1) % len(s['players'])]

    return s, reward, done, {"action": action}

# -------------------------
# Heuristic features (10 heuristics)
# -------------------------
def in_bounds(r,c):
    return 0<=r<ROWS and 0<=c<COLS

def count_in_direction(chips, r, c, dr, dc, team_map, team_id):
    """Count consecutive chips belonging to team_id starting from adjacent cell in direction."""
    cnt = 0
    rr, cc = r+dr, c+dc
    while in_bounds(rr,cc):
        val = chips[rr][cc]
        if (rr,cc) in CORNER_POS:
            cnt += 1
            rr += dr; cc += dc
            continue
        if val != 0 and team_map[val]==team_id:
            cnt += 1
            rr += dr; cc += dc
        else:
            break
    return cnt

def evaluate_action_features(state: dict, player_id: int, action: Action) -> Dict[str, float]:
    """
    Return a dict of feature values for the action for use in weighted scoring.
    Features correspond to the 10 heuristics (values normalized heuristically).
    """
    s = state
    board = s['board']; chips = s['chips']; team_map = s['player_to_team']
    player_team = team_map[player_id]
    typ, card, pos = action
    features = {f"f{i}":0.0 for i in range(1,11)}

    # if the action is a jack-one (remove), approximate by measuring how many opponent threats are removed
    if typ == "play_jack_one":
        r,c = pos
        owner_before = chips[r][c]
        # count opponent's adjacent threats removed
        val = 0
        for dr,dc in [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            if in_bounds(r+dr,c+dc):
                owner = chips[r+dr][c+dc]
                if owner != 0 and team_map[owner] != player_team:
                    val += 1
        features['f2'] = val  # blocking/opponent removal prioritized
        # minor positive for avoiding helping opponent
        features['f5'] = val * 0.2
        return features

    # For placement actions (play_card / two-eyed jack)
    if typ in ("play_card","play_jack_two"):
        r,c = pos
        # Heuristic 1: Multi-directional threats (number of directions where placing gives >=1 contiguous friendly chips)
        dirs = [(1,0),(0,1),(1,1),(-1,1)]
        multi = 0
        for dr,dc in dirs:
            forward = count_in_direction(chips,r,c,dr,dc,team_map,player_team)
            back = count_in_direction(chips,r,c,-dr,-dc,team_map,player_team)
            if forward + back >= 1:
                multi += 1
        features['f1'] = multi / 4.0

        # Heuristic 2: Block opponent threats early — measure max opponent contiguous length that this cell would block
        max_block = 0
        for opp in s['players']:
            if team_map[opp] == player_team: continue
            for dr,dc in dirs:
                # simulate if opponent had contiguous run that includes this pos
                # count opponent on both sides
                opp_forward = 0
                rr,cc = r+dr, c+dc
                while in_bounds(rr,cc):
                    owner = chips[rr][cc]
                    if owner!=0 and team_map[owner]!=player_team:
                        opp_forward += 1; rr+=dr; cc+=dc
                    else: break
                opp_back = 0
                rr,cc = r-dr, c-dc
                while in_bounds(rr,cc):
                    owner = chips[rr][cc]
                    if owner!=0 and team_map[owner]!=player_team:
                        opp_back += 1; rr-=dr; cc-=dc
                    else: break
                max_block = max(max_block, opp_forward+opp_back)
        features['f2'] = max_block / 5.0

        # Heuristic 3: Control center — inverse distance to board center
        center_r, center_c = (ROWS-1)/2.0, (COLS-1)/2.0
        dist = ((r-center_r)**2 + (c-center_c)**2)**0.5
        maxd = ((center_r)**2 + (center_c)**2)**0.5
        features['f3'] = max(0, 1 - dist / maxd)

        # Heuristic 4: Extend open-ended lines — count lines with space on both ends
        open_ended = 0
        for dr,dc in dirs:
            a_r, a_c = r+dr, c+dc
            b_r, b_c = r-dr, c-dc
            if in_bounds(a_r,a_c) and in_bounds(b_r,b_c):
                end_a_free = (chips[a_r][a_c]==0 or (a_r,a_c) in CORNER_POS)
                end_b_free = (chips[b_r][b_c]==0 or (b_r,b_c) in CORNER_POS)
                if end_a_free and end_b_free:
                    open_ended += 1
        features['f4'] = open_ended / 4.0

        # Heuristic 5: Avoid helping opponent — penalize if placing adjacent to opponent chips increases their local density
        help = 0
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                rr,cc = r+dr, c+dc
                if in_bounds(rr,cc):
                    owner = chips[rr][cc]
                    if owner !=0 and team_map[owner] != player_team:
                        help += 1
        features['f5'] = -help / 8.0

        # Heuristic 6: Create forced responses — placing that would create immediate 2-way threat (fork)
        forks = 0
        for dr,dc in dirs:
            forward = count_in_direction(chips,r,c,dr,dc,team_map,player_team)
            back = count_in_direction(chips,r,c,-dr,-dc,team_map,player_team)
            if forward+back >=3:  # this plus one would be 4 -> threat
                forks += 1
        features['f6'] = forks / 4.0

        # Heuristic 7: Build redundant paths (number of different near-complete lines created)
        redundant = 0
        for dr,dc in dirs:
            total = count_in_direction(chips,r,c,dr,dc,team_map,player_team) + count_in_direction(chips,r,c,-dr,-dc,team_map,player_team)
            if total >= 2:
                redundant += 1
        features['f7'] = redundant / 4.0

        # Heuristic 8: Exploit symmetry — check mirrored cell occupancy
        symr, symc = ROWS-1-r, COLS-1-c
        features['f8'] = 1.0 if in_bounds(symr,symc) and chips[symr][symc]!=0 and team_map[chips[symr][symc]]==player_team else 0.0

        # Heuristic 9: Quiet moves — latent threat potential (create 3-in-a-row with open ends)
        latent = 0
        for dr,dc in dirs:
            tot = count_in_direction(chips,r,c,dr,dc,team_map,player_team) + count_in_direction(chips,r,c,-dr,-dc,team_map,player_team)
            if tot == 2:
                # check ends open
                a_r, a_c = r+dr*(tot+1), c+dc*(tot+1)
                b_r, b_c = r-dr*(tot+1), c-dc*(tot+1)
                enda = in_bounds(a_r,a_c) and chips[a_r][a_c]==0
                endb = in_bounds(b_r,b_c) and chips[b_r][b_c]==0
                if enda or endb:
                    latent += 1
        features['f9'] = latent / 4.0

        # Heuristic 10: Local density (friendly minus opponent in 3x3 area)
        friendly = 0; opponent = 0
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                rr,cc = r+dr,c+dc
                if not in_bounds(rr,cc): continue
                val = chips[rr][cc]
                if val==0: continue
                if team_map[val]==player_team: friendly += 1
                else: opponent += 1
        features['f10'] = (friendly - opponent) / 9.0

    return features

# -------------------------
# Weighted scorer & AI
# -------------------------
DEFAULT_WEIGHTS = {
    'f1': 2.0,   # multi-directional threats
    'f2': 3.0,   # block opponent
    'f3': 0.8,   # center control
    'f4': 1.2,   # open-ended
    'f5': 1.5,   # avoid helping opponent (penalty)
    'f6': 2.2,   # forced responses
    'f7': 1.6,   # redundant paths
    'f8': 0.6,   # symmetry
    'f9': 1.0,   # quiet moves
    'f10': 1.0   # local density
}

def score_action(state: dict, player_id: int, action: Action, weights: Dict[str,float]) -> float:
    feats = evaluate_action_features(state, player_id, action)
    score = sum(feats[k]*weights.get(k,1.0) for k in feats)
    # small tie-breakers: prefer center positions slightly
    typ, card, pos = action
    if pos != (-1,-1):
        r,c = pos
        center_bonus = (ROWS/2 - abs(r-ROWS/2) + COLS/2 - abs(c-COLS/2)) / (ROWS+COLS)
        score += 0.01 * center_bonus
    return score

def choose_best_action(state: dict, player_id: int, weights: Dict[str,float]) -> Optional[Action]:
    actions = legal_actions(state, player_id)
    if not actions:
        return None
    best = None; best_score = -1e9
    for a in actions:
        sc = score_action(state, player_id, a, weights)
        if sc > best_score:
            best_score = sc; best = a
    return best

# -------------------------
# Simulator & UI
# -------------------------
def new_game(num_players:int, teams:Optional[Dict[int,int]]=None, seed:Optional[int]=None) -> dict:
    if seed is not None:
        random.seed(seed)
    board = make_official_board()
    deck = create_deck()
    hands = {p: [deck.pop(0) for _ in range(get_hand_size(num_players))] for p in range(1, num_players+1)}
    chips = [[0]*COLS for _ in range(ROWS)]
    completed = set()
    if teams is None:
        player_to_team = {p:p for p in range(1, num_players+1)}
    else:
        player_to_team = teams
    state = {
        'board': board,
        'deck': deck,
        'hands': hands,
        'chips': chips,
        'completed': completed,
        'discard': [],
        'player_to_team': player_to_team,
        'players': [p for p in range(1, num_players+1)],
        'current_player': 1,
        'seq_counts': {t:0 for t in set(player_to_team.values())}
    }
    return state

def game_step(state: dict, weights: Dict[str,float]) -> Tuple[dict, dict]:
    """
    Run a single move for the current player using heuristics and return (new_state, info)
    info contains action, score, time, etc.
    """
    player = state['current_player']
    action = choose_best_action(state, player, weights)
    if action is None:
        # no legal action: advance turn
        idx = state['players'].index(player)
        state['current_player'] = state['players'][(idx+1)%len(state['players'])]
        return state, {'action':None}
    sc = score_action(state, player, action, weights)
    new_state, reward, done, info = apply_action(state, player, action)
    info.update({'score': sc, 'player': player, 'reward': reward})
    return new_state, info

# -------------------------
# Batch runner for stats (no rendering)
# -------------------------
def run_batch(n_games: int, num_players: int, weights: Dict[str,float]):
    records = []
    for g in range(n_games):
        s = new_game(num_players)
        move_count = 0
        while True:
            s, info = game_step(s, weights)
            move_count += 1
            if any(v>= (2 if len(set(s['player_to_team'].values()))==2 else 1) for v in s['seq_counts'].values()):
                # determine winner team
                winner_team = max(s['seq_counts'].items(), key=lambda kv: kv[1])[0]
                records.append({'game': g, 'moves': move_count, 'winner_team': winner_team})
                break
            if move_count > 2000:  # abort runaway games
                records.append({'game': g, 'moves': move_count, 'winner_team': None})
                break
    return pd.DataFrame(records)

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(layout="wide", page_title="Sequence Research Simulator")

st.sidebar.header("Simulation Controls")
num_players = st.sidebar.selectbox("Number of players", options=[2,3,4,6,8,9,10,12], index=2)
player_mode = st.sidebar.selectbox("Mode", options=["GUI (play/observe)", "Batch (fast)"], index=0)
render_each_move = st.sidebar.checkbox("Render each move (slow)", value=True)
speed = st.sidebar.slider("Delay (s) between moves (GUI mode)", 0.0, 2.0, 0.4, 0.1)
batch_games = st.sidebar.number_input("Batch games (if Batch mode)", min_value=1, max_value=2000, value=200, step=10)

st.sidebar.header("Heuristic weights (adjust for experiments)")
weights = {}
for k,v in DEFAULT_WEIGHTS.items():
    weights[k] = st.sidebar.slider(k, -5.0, 10.0, float(v), 0.1)

st.sidebar.markdown("**Actions**")
if st.sidebar.button("Run one demo game (GUI)"):
    player_mode = "GUI (play/observe)"

if st.sidebar.button("Run batch (collect stats)"):
    player_mode = "Batch (fast)"

# main area
col1, col2 = st.columns([2,1])

with col1:
    st.header("Board")
    place_holder = st.empty()

with col2:
    st.header("Controls & Metrics")
    metrics_box = st.empty()
    log_box = st.expander("Move Log (last 200)", expanded=False)
    log_area = log_box.empty()

# Run modes
if player_mode.startswith("Batch"):
    st.write(f"Running batch of {batch_games} games (no rendering). This may take a moment...")
    df = run_batch(batch_games, num_players, weights)
    st.write("Batch complete — summary:")
    st.dataframe(df.describe(include='all'))
    st.download_button("Download batch results (CSV)", df.to_csv(index=False), file_name="sequence_batch.csv")
else:
    # GUI interactive demo — full rendering each move
    state = new_game(num_players)
    logs = []
    board_img_holder = place_holder.empty()
    # show initial board
    img = compose_board_image(state['board'], state['chips'], highlights=[])
    buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
    board_img_holder.image(buf)

    # show initial hands
    def show_hands(state):
        cols = st.columns(len(state['players']))
        for idx,p in enumerate(state['players']):
            cols[idx].markdown(f"**P{p} (team {state['player_to_team'][p]})**")
            # render small card images horizontally
            row_imgs=[]
            for card in state['hands'][p]:
                im = draw_card_image(card, chip_owner=None, highlight=False)
                row_imgs.append(im)
            # combine horizontally
            if row_imgs:
                total_w = sum(im.width for im in row_imgs) + (len(row_imgs)-1)*4
                out = Image.new("RGB", (total_w, CARD_H), color=(240,240,240))
                x=0
                for im in row_imgs:
                    out.paste(im, (x,0)); x+=im.width+4
                buf = io.BytesIO(); out.save(buf, format="PNG"); buf.seek(0)
                cols[idx].image(buf, use_column_width=True)
            else:
                cols[idx].write("No cards")

    show_hands(state)
    st.write("Starting auto-play (heuristic AI). Press Stop (browser) to abort.")

    move_limit = 2000
    move_count = 0
    start_time = time.time()
    while True:
        # render board
        img = compose_board_image(state['board'], state['chips'], highlights=[])
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        board_img_holder.image(buf)

        # step
        state, info = game_step(state, weights)
        move_count += 1
        logs.append(info)
        if len(logs) > 200: logs = logs[-200:]
        log_area.write(pd.DataFrame(logs[-20:]))

        # update hands display
        show_hands(state)

        # render updated board highlighting last action
        if info.get('action') and info['action'][2] != (-1,-1):
            last_pos = info['action'][2]
        else:
            last_pos = []
        img = compose_board_image(state['board'], state['chips'], highlights=[last_pos] if last_pos else [])
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        board_img_holder.image(buf)

        # update metrics
        metrics_box.metric("Moves", move_count)
        metrics_box.write("Seq counts: " + str(state['seq_counts']))

        if any(v>= (2 if len(set(state['player_to_team'].values()))==2 else 1) for v in state['seq_counts'].values()):
            st.success(f"Game ended in {move_count} moves. Seq counts: {state['seq_counts']}")
            break
        if move_count >= move_limit:
            st.warning("Move limit reached; stopping.")
            break

        if render_each_move:
            time.sleep(speed)
            continue
        else:
            # fast forward: execute many moves without rendering; break periodically to update UI
            for _ in range(10):
                state, info = game_step(state, weights)
                move_count += 1
                logs.append(info)
                if len(logs) > 200: logs = logs[-200:]
                if any(v>= (2 if len(set(state['player_to_team'].values()))==2 else 1) for v in state['seq_counts'].values()):
                    break
            # update minimal UI
            log_area.write(pd.DataFrame(logs[-20:]))
            metrics_box.metric("Moves", move_count)
            metrics_box.write("Seq counts: " + str(state['seq_counts']))

    # after game: produce move log download
    df_logs = pd.DataFrame(logs)
    csv = df_logs.to_csv(index=False).encode('utf-8')
    st.download_button("Download move log (CSV)", data=csv, file_name="sequence_move_log.csv", mime="text/csv")
