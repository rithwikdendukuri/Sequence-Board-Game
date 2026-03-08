# sequence_simulator_final.py
"""
Sequence Simulator — Final polished interactive version.
Features:
 - Full official Sequence rules (2 decks, corner wilds, one-eyed/two-eyed jacks, dead cards)
 - Human vs AI (clickable hand + clickable board with highlighted valid targets)
 - AI vs AI (step / auto-play with speed control)
 - "Nerd Stuff" tab for adjusting heuristic weights and debugging
 - Move history, last-move highlights, completed sequence locks
 - Face cards show stylized royals; numeric cards show the suit glyph; corners show a jester icon
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import random, time, io
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Set

# -------------------------
# Config / Constants
# -------------------------
ROWS, COLS = 10, 10
CARD_W, CARD_H = 72, 100
PADDING = 6
BOARD_BG = (24, 80, 40)
PLAYER_COLORS = [(220, 40, 40), (40, 120, 220), (220, 180, 40), (120, 40, 180),
                 (30, 180, 100), (140, 90, 30), (60,60,60), (180, 60, 100)]
ONE_EYED = {"J♠", "J♥"}    # removes opponent chip
TWO_EYED = {"J♣", "J♦"}    # place anywhere open
CORNER_POS = {(0,0),(0,9),(9,0),(9,9)}
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]
STANDARD_DECK = [f"{r}{s}" for r in RANKS for s in SUITS]  # 52 unique ids
HAND_SIZE_MAP = {2:7,3:6,4:6,6:5,8:4,9:4,10:3,12:3}

# -------------------------
# Fonts & drawing helpers
# -------------------------
def _get_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

FONT_SMALL = _get_font(12)
FONT_MED = _get_font(16)
FONT_LARGE = _get_font(22)
FONT_ROYAL = _get_font(28)
FONT_JESTER = _get_font(36)

# draw_card_image supports highlight_type: None | "valid" | "last" | "selected" | "completed"
@st.cache_data(show_spinner=False)
def draw_card_image_bytes(card: str,
                          chip_owner: Optional[int]=None,
                          highlight_type: Optional[str]=None,
                          corner_wild: bool=False) -> bytes:
    """
    Return PNG bytes of a card cell. highlight_type controls border color:
      - valid -> green border (valid target)
      - last  -> gold border (last move)
      - selected -> cyan border (selected by human)
      - completed -> blue border (locked sequence)
    """
    img = Image.new("RGB", (CARD_W, CARD_H), color="white")
    draw = ImageDraw.Draw(img)

    # choose border color
    if highlight_type == "valid":
        border_color = (40, 180, 60)
    elif highlight_type == "last":
        border_color = (255, 215, 60)
    elif highlight_type == "selected":
        border_color = (60, 200, 200)
    elif highlight_type == "completed":
        border_color = (60, 120, 200)
    else:
        border_color = (200,200,200)

    draw.rectangle([1,1,CARD_W-2,CARD_H-2], outline=border_color, width=3 if highlight_type else 2)

    if card == "WILD" or corner_wild:
        # jester emoji in center if available, otherwise "JEST"
        try:
            draw.text((CARD_W//2, CARD_H//2), "🤹", anchor="mm", font=FONT_JESTER)
        except Exception:
            draw.text((CARD_W//2, CARD_H//2), "JEST", anchor="mm", font=FONT_MED)
        # golden accent
        draw.rectangle([4,20,CARD_W-4,CARD_H-12], outline=(230,190,30), width=2)
    else:
        rank = card[:-1]
        suit = card[-1]
        suit_color = (0,0,0) if suit in ["♠","♣"] else (180,0,0)

        # corners for rank and suit
        draw.text((6,6), rank, font=FONT_SMALL, fill=suit_color)
        draw.text((CARD_W-10, CARD_H-18), suit, font=FONT_SMALL, fill=suit_color, anchor="rm")

        if rank in ["J","Q","K"]:
            # stylized royal box + rank + small crown/icon
            rr = (CARD_W//2-20, CARD_H//2-28, CARD_W//2+20, CARD_H//2+28)
            draw.rectangle(rr, outline=suit_color, width=2)
            draw.text((CARD_W//2, CARD_H//2-6), rank, anchor="mm", font=FONT_ROYAL, fill=suit_color)
            if rank == "K":
                draw.text((CARD_W//2, CARD_H//2-32), "♔", anchor="mm", font=FONT_SMALL, fill=suit_color)
            elif rank == "Q":
                draw.text((CARD_W//2, CARD_H//2-32), "♕", anchor="mm", font=FONT_SMALL, fill=suit_color)
            else:
                draw.text((CARD_W//2, CARD_H//2-32), "♪", anchor="mm", font=FONT_SMALL, fill=suit_color)
        else:
            # numeric/ace: show suit glyph prominently
            draw.text((CARD_W//2, CARD_H//2-6), suit, anchor="mm", font=FONT_LARGE, fill=suit_color)

        draw.text((6, CARD_H-18), rank, font=FONT_SMALL, fill=suit_color)

    # chip overlay
    if chip_owner is not None and chip_owner > 0:
        color = PLAYER_COLORS[(chip_owner-1) % len(PLAYER_COLORS)]
        cx, cy, r = CARD_W-18, 18, 12
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color, outline=(0,0,0))
        draw.text((cx, cy), str(chip_owner), anchor="mm", font=FONT_SMALL, fill=(255,255,255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# -------------------------
# Board & deck generation
# -------------------------
def make_official_board(seed: Optional[int]=None) -> List[List[str]]:
    if seed is not None:
        random.seed(seed)
    non_jacks = [c for c in STANDARD_DECK if not c.startswith("J")]
    tokens = non_jacks * 2  # 96 tokens
    random.shuffle(tokens)
    grid = [["" for _ in range(COLS)] for _ in range(ROWS)]
    idx = 0
    for r in range(ROWS):
        for c in range(COLS):
            if (r,c) in CORNER_POS:
                grid[r][c] = "WILD"
            else:
                grid[r][c] = tokens[idx]; idx += 1
    return grid

def create_deck() -> List[str]:
    deck = STANDARD_DECK.copy() + STANDARD_DECK.copy()
    random.shuffle(deck)
    return deck

def get_hand_size(num_players: int) -> int:
    return HAND_SIZE_MAP.get(num_players, 6)

# -------------------------
# Sequence detection & rules
# -------------------------
def in_bounds(r,c): return 0 <= r < ROWS and 0 <= c < COLS

def find_sequence_positions(chips: List[List[int]], mapping: Dict[int,int], team_id: int, min_len: int=5) -> Set[Tuple[int,int]]:
    dirs = [(0,1),(1,0),(1,1),(-1,1)]
    seqpos = set()
    for r in range(ROWS):
        for c in range(COLS):
            if not (((r,c) in CORNER_POS) or (chips[r][c]!=0 and mapping.get(chips[r][c])==team_id)):
                continue
            for dr,dc in dirs:
                pr,pc = r-dr, c-dc
                if in_bounds(pr,pc) and (((pr,pc) in CORNER_POS) or (chips[pr][pc]!=0 and mapping.get(chips[pr][pc])==team_id)):
                    continue
                run=[]
                rr,cc = r,c
                while in_bounds(rr,cc) and (((rr,cc) in CORNER_POS) or (chips[rr][cc]!=0 and mapping.get(chips[rr][cc])==team_id)):
                    run.append((rr,cc))
                    rr += dr; cc += dc
                if len(run) >= min_len:
                    seqpos.update(run)
    return seqpos

def count_team_sequences(chips: List[List[int]], mapping: Dict[int,int]) -> Dict[int,int]:
    teams = set(mapping.values())
    out = {}
    for t in teams:
        pos = find_sequence_positions(chips, mapping, t)
        out[t] = len(pos) // 5
    return out

# -------------------------
# Actions & state transitions
# -------------------------
Action = Tuple[str, str, Tuple[int,int]]  # type, card, pos

def positions_of_card_on_board(board: List[List[str]], card: str) -> List[Tuple[int,int]]:
    return [(r,c) for r in range(ROWS) for c in range(COLS) if board[r][c]==card]

def legal_actions(state: dict, player_id: int) -> List[Action]:
    board = state['board']; chips = state['chips']; hands = state['hands']; completed = state['completed']; mapping = state['player_to_team']
    hand = hands[player_id]
    actions = []
    for card in hand:
        if card in TWO_EYED:
            for r in range(ROWS):
                for c in range(COLS):
                    if (r,c) in CORNER_POS or (r,c) in completed: continue
                    if chips[r][c]==0:
                        actions.append(("play_jack_two", card, (r,c)))
        elif card in ONE_EYED:
            for r in range(ROWS):
                for c in range(COLS):
                    if (r,c) in CORNER_POS or (r,c) in completed: continue
                    owner = chips[r][c]
                    if owner != 0 and owner != player_id:
                        actions.append(("play_jack_one", card, (r,c)))
        else:
            poss = positions_of_card_on_board(board, card)
            placed_any = False
            for (r,c) in poss:
                if chips[r][c]==0 and (r,c) not in completed and (r,c) not in CORNER_POS:
                    actions.append(("play_card", card, (r,c)))
                    placed_any = True
            if not placed_any:
                actions.append(("turn_in_dead", card, (-1,-1)))
    return actions

def reshuffle_discard_to_deck(s: dict):
    if s['discard']:
        s['deck'].extend(s['discard'])
        s['discard'].clear()
        random.shuffle(s['deck'])

def draw_card_safe(s: dict) -> Optional[str]:
    if not s['deck'] and s['discard']:
        reshuffle_discard_to_deck(s)
    if not s['deck']:
        return None
    return s['deck'].pop(0)

def apply_action(state: dict, player_id: int, action: Action, draw_after: bool=True):
    s = deepcopy(state)
    typ, card, pos = action
    chips = s['chips']; completed = s['completed']; mapping = s['player_to_team']
    team_id = mapping[player_id]
    prev_team_seq = count_team_sequences(chips, mapping).get(team_id, 0)

    if typ == "play_card":
        r,c = pos
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
        s['hands'][player_id].remove(card)
        s['discard'].append(card)
        chips[r][c] = 0
    elif typ == "turn_in_dead":
        s['hands'][player_id].remove(card)
        s['discard'].append(card)
    else:
        raise ValueError("unknown action type")

    if draw_after:
        drawn = draw_card_safe(s)
        if drawn:
            s['hands'][player_id].append(drawn)

    seqpos = find_sequence_positions(chips, mapping, team_id)
    if seqpos:
        s['completed'].update(seqpos)
    s['seq_counts'] = count_team_sequences(chips, mapping)

    num_teams = len(set(mapping.values()))
    required = 2 if num_teams == 2 else 1
    done = any(v >= required for v in s['seq_counts'].values())

    new_team_seq = s['seq_counts'].get(team_id, 0)
    reward = max(0, new_team_seq - prev_team_seq)

    idx = s['players'].index(player_id)
    s['current_player'] = s['players'][(idx+1) % len(s['players'])]

    return s, reward, done, {"action":action}

# -------------------------
# Heuristics & AI
# -------------------------
DEFAULT_WEIGHTS = {
    'f1': 2.0, 'f2': 3.0, 'f3': 0.8, 'f4': 1.2, 'f5': 1.5,
    'f6': 2.2, 'f7': 1.6, 'f8': 0.6, 'f9': 1.0, 'f10': 1.0
}

def in_bounds_local(r,c):
    return 0<=r<ROWS and 0<=c<COLS

def count_in_direction(chips, r, c, dr, dc, mapping, team_id):
    cnt=0; rr,cc=r+dr,c+dc
    while in_bounds_local(rr,cc):
        if (rr,cc) in CORNER_POS:
            cnt += 1; rr+=dr; cc+=dc; continue
        val = chips[rr][cc]
        if val != 0 and mapping[val] == team_id:
            cnt += 1; rr+=dr; cc+=dc
        else: break
    return cnt

def evaluate_action_features(state: dict, player_id: int, action: Action) -> Dict[str,float]:
    s = state; board = s['board']; chips = s['chips']; mapping = s['player_to_team']
    player_team = mapping[player_id]
    typ, card, pos = action
    feats = {f"f{i}":0.0 for i in range(1,11)}

    if typ == "play_jack_one":
        r,c = pos
        val = 0
        for dr,dc in [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            rr,cc = r+dr, c+dc
            if in_bounds_local(rr,cc):
                owner = chips[rr][cc]
                if owner != 0 and mapping[owner] != player_team:
                    val += 1
        feats['f2'] = val
        return feats

    if typ in ("play_card","play_jack_two"):
        r,c = pos
        dirs = [(1,0),(0,1),(1,1),(-1,1)]
        multi = 0
        for dr,dc in dirs:
            forward = count_in_direction(chips,r,c,dr,dc,mapping,player_team)
            back = count_in_direction(chips,r,c,-dr,-dc,mapping,player_team)
            if forward + back >= 1: multi += 1
        feats['f1'] = multi/4.0

        max_block = 0
        for opp in s['players']:
            if mapping[opp] == player_team: continue
            for dr,dc in dirs:
                opp_forward=0; rr,cc=r+dr,c+dc
                while in_bounds_local(rr,cc):
                    owner = chips[rr][cc]
                    if owner!=0 and mapping[owner]!=player_team:
                        opp_forward+=1; rr+=dr; cc+=dc
                    else: break
                opp_back=0; rr,cc=r-dr,c-dc
                while in_bounds_local(rr,cc):
                    owner = chips[rr][cc]
                    if owner!=0 and mapping[owner]!=player_team:
                        opp_back+=1; rr-=dr; cc-=dc
                    else: break
                max_block = max(max_block, opp_forward+opp_back)
        feats['f2'] = max_block / 5.0

        center_r,center_c=(ROWS-1)/2.0,(COLS-1)/2.0
        dist = ((r-center_r)**2 + (c-center_c)**2)**0.5
        maxd = ((center_r)**2 + (center_c)**2)**0.5
        feats['f3'] = max(0, 1 - dist/maxd)

        open_ended=0
        for dr,dc in dirs:
            a_r,a_c = r+dr,c+dc; b_r,b_c = r-dr,c-dc
            if in_bounds_local(a_r,a_c) and in_bounds_local(b_r,b_c):
                end_a_free = (chips[a_r][a_c]==0 or (a_r,a_c) in CORNER_POS)
                end_b_free = (chips[b_r][b_c]==0 or (b_r,b_c) in CORNER_POS)
                if end_a_free and end_b_free: open_ended+=1
        feats['f4']=open_ended/4.0

        helpv=0
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                rr,cc=r+dr,c+dc
                if in_bounds_local(rr,cc):
                    owner = chips[rr][cc]
                    if owner !=0 and mapping[owner] != player_team: helpv+=1
        feats['f5'] = -helpv/8.0

        forks=0
        for dr,dc in dirs:
            forward = count_in_direction(chips,r,c,dr,dc,mapping,player_team)
            back = count_in_direction(chips,r,c,-dr,-dc,mapping,player_team)
            if forward+back >= 3: forks+=1
        feats['f6'] = forks / 4.0

        redundant = 0
        for dr,dc in dirs:
            total = count_in_direction(chips,r,c,dr,dc,mapping,player_team)+count_in_direction(chips,r,c,-dr,-dc,mapping,player_team)
            if total >= 2: redundant += 1
        feats['f7'] = redundant/4.0

        symr,symc = ROWS-1-r, COLS-1-c
        feats['f8'] = 1.0 if in_bounds_local(symr,symc) and chips[symr][symc]!=0 and mapping[chips[symr][symc]]==player_team else 0.0

        latent=0
        for dr,dc in dirs:
            tot = count_in_direction(chips,r,c,dr,dc,mapping,player_team)+count_in_direction(chips,r,c,-dr,-dc,mapping,player_team)
            if tot==2:
                a_r,a_c = r+dr*(tot+1), c+dc*(tot+1)
                b_r,b_c = r-dr*(tot+1), c-dc*(tot+1)
                enda = in_bounds_local(a_r,a_c) and chips[a_r][a_c]==0
                endb = in_bounds_local(b_r,b_c) and chips[b_r][b_c]==0
                if enda or endb: latent += 1
        feats['f9'] = latent/4.0

        friendly=0; opponent=0
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                rr,cc = r+dr,c+dc
                if not in_bounds_local(rr,cc): continue
                val = chips[rr][cc]
                if val==0: continue
                if mapping[val]==player_team: friendly += 1
                else: opponent += 1
        feats['f10'] = (friendly - opponent)/9.0

    return feats

def score_action(state: dict, player_id: int, action: Action, weights: Dict[str,float]) -> float:
    feats = evaluate_action_features(state, player_id, action)
    score = sum(feats[k] * weights.get(k, 1.0) for k in feats)
    typ, card, pos = action
    if pos != (-1,-1):
        r,c = pos
        center_bonus = (ROWS/2 - abs(r-ROWS/2) + COLS/2 - abs(c-COLS/2)) / (ROWS+COLS)
        score += 0.01 * center_bonus
    return score

def choose_best_action_and_explain(state: dict, player_id: int, weights: Dict[str,float]) -> Tuple[Optional[Action], Optional[str]]:
    actions = legal_actions(state, player_id)
    if not actions:
        return None, "No legal actions available."
    best=None; best_score=-1e9
    best_feats=None
    for a in actions:
        sc = score_action(state, player_id, a, weights)
        if sc > best_score:
            best_score = sc; best = a
    feats = evaluate_action_features(state, player_id, best)
    contribs = {k: feats[k]*weights.get(k,1.0) for k in feats}
    sorted_contribs = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
    mapping_names = {
        'f1': "multi-directional threat",
        'f2': "blocking opponent",
        'f3': "center control",
        'f4': "open-ended",
        'f5': "avoid helping opponent",
        'f6': "create forced responses (fork)",
        'f7': "redundant paths",
        'f8': "symmetry",
        'f9': "latent 3-in-row threat",
        'f10': "local friendly density"
    }
    reasons = []
    for k,v in sorted_contribs[:2]:
        reasons.append(f"{mapping_names.get(k,k)} ({v:.2f})")
    explanation = f"Top reasons: {', '.join(reasons)}"
    return best, explanation

# -------------------------
# New game / simulator helpers
# -------------------------
def new_game(num_players:int, teams:Optional[Dict[int,int]]=None, seed:Optional[int]=None) -> dict:
    if seed is not None: random.seed(seed)
    board = make_official_board()
    deck = create_deck()
    hands = {p: [deck.pop(0) for _ in range(get_hand_size(num_players))] for p in range(1, num_players+1)}
    chips = [[0]*COLS for _ in range(ROWS)]
    completed = set()
    player_to_team = {p:p for p in range(1, num_players+1)} if teams is None else teams
    state = {
        'board': board, 'deck': deck, 'hands': hands, 'chips': chips, 'completed': completed,
        'discard': [], 'player_to_team': player_to_team, 'players': [p for p in range(1, num_players+1)],
        'current_player': 1, 'seq_counts': {t:0 for t in set(player_to_team.values())}
    }
    return state

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide", page_title="Sequence Simulator — Pro")

st.title("Sequence Simulator — Pro (Rules-complete, Human vs AI & AI vs AI)")

# top controls area
top_left, top_right = st.columns([3,1])
with top_right:
    num_players = st.selectbox("Number of players", [2,3,4,6,8,9,10,12], index=0)
    mode = st.selectbox("Mode", ["Human vs AI","AI vs AI"], index=0)
    start_btn = st.button("New Game (Reset)")
    auto_speed = st.slider("AI speed (s per move)", 0.0, 1.0, 0.25, 0.05)

# nerd tab
with st.expander("🧠 Nerd Stuff (Heuristics & Debug)", expanded=False):
    st.markdown("Adjust AI heuristic weights (affects AI choices).")
    weight_sliders = {}
    for k,v in DEFAULT_WEIGHTS.items():
        weight_sliders[k] = st.slider(k, -5.0, 10.0, float(v), 0.1, key=f"w_{k}")
    show_ai_debug = st.checkbox("Show AI candidate scores", value=False)

# initialize session state
if 'game' not in st.session_state or start_btn:
    st.session_state.game = new_game(num_players)
    st.session_state.mode = mode
    st.session_state.selected_card_idx = None
    st.session_state.valid_targets = set()
    st.session_state.last_move = None
    st.session_state.history = [deepcopy(st.session_state.game)]
    st.session_state.history_ptr = 0
    st.session_state.auto_play = False
    st.session_state.ai_thinking = ""
    st.session_state.message = "New game started."
    st.session_state.weights = DEFAULT_WEIGHTS.copy()
    # apply sliders if nerd tab was used
    try:
        st.session_state.weights.update(weight_sliders)
    except Exception:
        pass

# sync weights from sliders if present
try:
    if 'weight_sliders' in locals():
        st.session_state.weights.update(weight_sliders)
except Exception:
    pass

game = st.session_state.game

# helper: compute valid targets for a selected card (only for human player 1)
def compute_valid_targets_for_card(game_state: dict, player_id: int, card_idx: int) -> Set[Tuple[int,int]]:
    valid = set()
    if card_idx is None: return valid
    hands = game_state['hands']
    if player_id not in hands: return valid
    if card_idx < 0 or card_idx >= len(hands[player_id]): return valid
    card = hands[player_id][card_idx]
    if card in TWO_EYED:
        for r in range(ROWS):
            for c in range(COLS):
                if (r,c) in CORNER_POS or (r,c) in game_state['completed']: continue
                if game_state['chips'][r][c] == 0:
                    valid.add((r,c))
    elif card in ONE_EYED:
        for r in range(ROWS):
            for c in range(COLS):
                if (r,c) in CORNER_POS or (r,c) in game_state['completed']: continue
                owner = game_state['chips'][r][c]
                if owner != 0 and owner != player_id:
                    valid.add((r,c))
    else:
        poss = positions_of_card_on_board(game_state['board'], card)
        for (r,c) in poss:
            if (r,c) in CORNER_POS or (r,c) in game_state['completed']: continue
            if game_state['chips'][r][c] == 0:
                valid.add((r,c))
    return valid

# UI Layout: board + right-side controls
board_col, right_col = st.columns([3,1])

with right_col:
    st.subheader("Game Info")
    st.markdown(f"**Mode:** {mode}")
    st.markdown(f"**Current Player:** P{game['current_player']} (team {game['player_to_team'][game['current_player']]})")
    st.markdown(f"**Sequences:** {game['seq_counts']}")
    st.write("---")
    st.markdown("**Controls**")
    if mode == "AI vs AI":
        if st.button("Step one move"):
            st.session_state.auto_play = False
            st.session_state._run_one = True
        if st.button("Auto-play (start/stop)"):
            st.session_state.auto_play = not st.session_state.auto_play
    else:
        # human mode controls
        st.write("Human controls (Player 1): select a card then click a highlighted cell to play.")
        if st.button("Turn In Selected Dead Card"):
            sel = st.session_state.selected_card_idx
            if sel is None:
                st.session_state.message = "No card selected."
            else:
                # validate dead
                card = game['hands'][1][sel]
                if card.startswith("J"):
                    st.session_state.message = "Jacks cannot be turned in."
                else:
                    poss = positions_of_card_on_board(game['board'], card)
                    both_blocked = True
                    for (r,c) in poss:
                        if game['chips'][r][c]==0 and (r,c) not in game['completed'] and (r,c) not in CORNER_POS:
                            both_blocked = False
                    if not both_blocked:
                        st.session_state.message = "Selected card is not dead."
                    else:
                        card_removed = game['hands'][1].pop(sel)
                        game['discard'].append(card_removed)
                        drawn = draw_card_safe(game)
                        if drawn:
                            game['hands'][1].append(drawn)
                        st.session_state.message = f"Turned in {card_removed}."
                        st.session_state.selected_card_idx = None
                        st.session_state.valid_targets = set()
                        st.session_state.history.append(deepcopy(game))
                        st.session_state.history_ptr = len(st.session_state.history)-1

    st.write("---")
    st.markdown("**AI thinking**")
    st.write(st.session_state.ai_thinking or "—")
    st.write("---")
    st.markdown("**Move log**")
    if 'move_log' not in st.session_state:
        st.session_state.move_log = []
    if st.button("Clear Log"):
        st.session_state.move_log = []

with board_col:
    st.subheader("Board")
    # render clickable grid row by row
    # Determine valid highlight sets
    if mode == "Human vs AI":
        st.session_state.valid_targets = compute_valid_targets_for_card(game, 1, st.session_state.selected_card_idx)
    else:
        st.session_state.valid_targets = set()

    last_move = st.session_state.last_move  # tuple (r,c) or None

    # Render board as grid of images + small transparent button overlay
    for r in range(ROWS):
        cols = st.columns(COLS)
        for c, col in enumerate(cols):
            card = game['board'][r][c]
            chip_owner = game['chips'][r][c]
            corner = (r,c) in CORNER_POS
            highlight_type = None
            if (r,c) in st.session_state.valid_targets:
                highlight_type = "valid"
            elif st.session_state.last_move == (r,c):
                highlight_type = "last"
            elif (r,c) in game['completed']:
                highlight_type = "completed"
            # create image bytes
            img_bytes = draw_card_image_bytes(card, chip_owner=chip_owner, highlight_type=highlight_type, corner_wild=corner)
            col.image(img_bytes, use_column_width=True)
            key = f"cell_{r}_{c}"
            if col.button("Play", key=key):
                # cell clicked: behavior depends on mode & whose turn
                current = game['current_player']
                if mode == "Human vs AI" and current == 1:
                    sel_idx = st.session_state.selected_card_idx
                    if sel_idx is None:
                        st.session_state.message = "Select a card from your hand first."
                    else:
                        card_to_play = game['hands'][1][sel_idx]
                        # validate action for this exact cell
                        valid = False; action = None
                        if card_to_play in TWO_EYED:
                            if (r,c) in CORNER_POS:
                                st.session_state.message = "Cannot place on corner (wild)."
                            elif (r,c) in game['completed']:
                                st.session_state.message = "Cannot place on completed sequence."
                            elif game['chips'][r][c] != 0:
                                st.session_state.message = "Space occupied."
                            else:
                                valid = True; action = ("play_jack_two", card_to_play, (r,c))
                        elif card_to_play in ONE_EYED:
                            if (r,c) in CORNER_POS:
                                st.session_state.message = "Cannot remove corner."
                            elif (r,c) in game['completed']:
                                st.session_state.message = "Cannot remove chip from completed sequence."
                            elif game['chips'][r][c] == 0:
                                st.session_state.message = "No chip to remove there."
                            elif game['chips'][r][c] == 1:
                                st.session_state.message = "Cannot remove your own chip."
                            else:
                                valid = True; action = ("play_jack_one", card_to_play, (r,c))
                        else:
                            if game['board'][r][c] != card_to_play:
                                st.session_state.message = "Card does not match that space."
                            elif (r,c) in CORNER_POS:
                                st.session_state.message = "Cannot place on corner (wild)."
                            elif (r,c) in game['completed']:
                                st.session_state.message = "Cannot place on completed sequence."
                            elif game['chips'][r][c] != 0:
                                st.session_state.message = "Space occupied."
                            else:
                                valid = True; action = ("play_card", card_to_play, (r,c))
                        if valid and action:
                            game_after, reward, done, info = apply_action(game, 1, action)
                            st.session_state.ai_thinking = ""
                            st.session_state.last_move = action[2] if action[2] != (-1,-1) else None
                            st.session_state.move_log.append(f"P1 played {action[1]} at {action[2]}")
                            game = game_after
                            st.session_state.game = game
                            st.session_state.history = st.session_state.history[:st.session_state.history_ptr+1] + [deepcopy(game)]
                            st.session_state.history_ptr = len(st.session_state.history)-1
                            st.session_state.selected_card_idx = None
                            st.session_state.valid_targets = set()
                            st.session_state.message = f"You played {action[1]} at {action[2]}."
                            if done:
                                st.success(f"Game ended. Sequences: {game['seq_counts']}")
                            # after successful human move, let AI move (one or auto depending)
                            break
                else:
                    # If AI vs AI or it's an AI's turn, the cell button tries to force a play (for debugging / manual override)
                    forced=False
                    for card in list(game['hands'][current]):
                        if card in TWO_EYED and game['chips'][r][c]==0 and (r,c) not in game['completed'] and (r,c) not in CORNER_POS:
                            act=("play_jack_two",card,(r,c)); forced=True; break
                        if card in ONE_EYED and game['chips'][r][c]!=0 and game['chips'][r][c]!=current and (r,c) not in game['completed'] and (r,c) not in CORNER_POS:
                            act=("play_jack_one",card,(r,c)); forced=True; break
                        if game['board'][r][c]==card and game['chips'][r][c]==0 and (r,c) not in game['completed'] and (r,c) not in CORNER_POS:
                            act=("play_card",card,(r,c)); forced=True; break
                    if forced:
                        game_after, reward, done, info = apply_action(game, current, act)
                        st.session_state.move_log.append(f"P{current} forced-play {act[1]} at {act[2]}")
                        game = game_after
                        st.session_state.game = game
                        st.session_state.history.append(deepcopy(game))
                        st.session_state.history_ptr = len(st.session_state.history)-1
                        st.session_state.message = f"P{current} forced-played {act[1]} at {act[2]}"
                    else:
                        st.session_state.message = "No forced-play possible for that cell."

        # end board grid rows

# right column continued: human hand and controls
with right_col:
    st.write("---")
    st.subheader("Player 1 Hand (click to select)")
    hand = game['hands'][1]
    if not hand:
        st.write("No cards.")
    else:
        cols_hand = st.columns(len(hand))
        for idx, card in enumerate(list(hand)):
            with cols_hand[idx]:
                b = draw_card_image_bytes(card, chip_owner=None, highlight_type=("selected" if st.session_state.selected_card_idx==idx else None))
                if st.button(card, key=f"hand_{idx}"):
                    # select/deselect
                    if st.session_state.selected_card_idx == idx:
                        st.session_state.selected_card_idx = None
                        st.session_state.valid_targets = set()
                    else:
                        st.session_state.selected_card_idx = idx
                        st.session_state.valid_targets = compute_valid_targets_for_card(game, 1, idx)
                st.image(b, use_column_width=True)

    st.write("---")
    st.write("Selected:", st.session_state.selected_card_idx, " — ", (hand[st.session_state.selected_card_idx] if st.session_state.selected_card_idx is not None and st.session_state.selected_card_idx < len(hand) else "None"))
    st.write("Message:", st.session_state.get("message",""))
    st.write("---")
    st.subheader("Move History / Replay")
    if st.button("Step Back"):
        if st.session_state.history_ptr > 0:
            st.session_state.history_ptr -= 1
            st.session_state.game = deepcopy(st.session_state.history[st.session_state.history_ptr])
            game = st.session_state.game
            st.session_state.message = "Stepped back."
    if st.button("Step Forward"):
        if st.session_state.history_ptr < len(st.session_state.history)-1:
            st.session_state.history_ptr += 1
            st.session_state.game = deepcopy(st.session_state.history[st.session_state.history_ptr])
            game = st.session_state.game
            st.session_state.message = "Stepped forward."
    if st.button("Reset to current game state"):
        st.session_state.history = [deepcopy(st.session_state.game)]
        st.session_state.history_ptr = 0
        st.session_state.message = "History reset."

# AI decision & autoplay loop
# Use current weights from session
weights = st.session_state.weights

def ai_play_one_turn():
    global game
    current = game['current_player']
    action, explanation = choose_best_action_and_explain(game, current, weights)
    if action is None:
        st.session_state.ai_thinking = "AI had no legal action."
        return False
    st.session_state.ai_thinking = explanation
    # small delay for UX
    time.sleep(min(0.6, auto_speed))
    game_after, reward, done, info = apply_action(game, current, action)
    st.session_state.last_move = action[2] if action[2] != (-1,-1) else None
    st.session_state.move_log.append(f"P{current} (AI) {action[0]} {action[1]} -> {action[2]}")
    # update
    st.session_state.game = game_after
    game = game_after
    # snapshot history
    st.session_state.history = st.session_state.history[:st.session_state.history_ptr+1] + [deepcopy(game)]
    st.session_state.history_ptr = len(st.session_state.history)-1
    if done:
        st.success(f"Game ended. Sequences: {game['seq_counts']}")
        return True
    return False

# decide whether to run AI move(s)
if mode == "Human vs AI":
    # if it's AI's turn, do a single AI move now (unless user wants to step)
    if game['current_player'] != 1:
        ended = ai_play_one_turn()
        if not ended:
            st.session_state.message = f"AI (P{game['current_player']-1}) moved. Your turn."
else:
    # AI vs AI mode: either one-step or autoplay loop
    if st.session_state.auto_play:
        # loop until done or until user stops (we update UI between iterations)
        loop_limit = 1000
        for _ in range(loop_limit):
            ended = ai_play_one_turn()
            time.sleep(auto_speed)
            if ended or not st.session_state.auto_play:
                break
    else:
        # if a manual step triggered
        if st.session_state.get("_run_one", False):
            st.session_state._run_one = False
            ai_play_one_turn()

# re-render final big board at bottom (overview)
st.write("---")
st.subheader("Board Overview")
big_w = COLS*(CARD_W+PADDING)+PADDING
big_h = ROWS*(CARD_H+PADDING)+PADDING
bg = Image.new("RGB", (big_w, big_h), color=BOARD_BG)
y0 = PADDING
for r in range(ROWS):
    x0 = PADDING
    for c in range(COLS):
        card = game['board'][r][c]
        chip_owner = game['chips'][r][c]
        corner = (r,c) in CORNER_POS
        highlight_type = None
        if (r,c) in st.session_state.valid_targets:
            highlight_type = "valid"
        elif st.session_state.last_move == (r,c):
            highlight_type = "last"
        elif (r,c) in game['completed']:
            highlight_type = "completed"
        img_bytes = draw_card_image_bytes(card, chip_owner=chip_owner, highlight_type=highlight_type, corner_wild=corner)
        img = Image.open(io.BytesIO(img_bytes))
        bg.paste(img, (x0,y0))
        x0 += CARD_W + PADDING
    y0 += CARD_H + PADDING
buf = io.BytesIO(); bg.save(buf, format="PNG"); buf.seek(0)
st.image(buf, use_column_width=True)

# final diagnostics & logs
st.write("---")
st.subheader("Game State & Logs")
cols_info = st.columns([2,1])
with cols_info[0]:
    st.write("Player hands sizes and deck/discard:")
    for p in game['players']:
        st.write(f"P{p}: {len(game['hands'][p])} cards — Team {game['player_to_team'][p]}")
with cols_info[1]:
    st.write(f"Deck: {len(game['deck'])}, Discard: {len(game['discard'])}")
st.write("Move log (recent):")
for m in st.session_state.move_log[-50:]:
    st.write(m)

# AI debugging table if requested
if show_ai_debug:
    st.write("AI candidate moves for current player")
    cur = game['current_player']
    acts = legal_actions(game, cur)
    scored = []
    for a in acts:
        sc = score_action(game, cur, a, st.session_state.weights)
        feats = evaluate_action_features(game, cur, a)
        scored.append((a, sc, feats))
    scored.sort(key=lambda x: x[1], reverse=True)
    for a, sc, feats in scored[:30]:
        st.write(f"{a} => {sc:.3f} ; feats: {feats}")

st.caption("Rules enforced: corners wild (do not place), one-eyed removes (not corners/completed), two-eyed places anywhere open, dead-card turn-in allowed at start of your turn.")
