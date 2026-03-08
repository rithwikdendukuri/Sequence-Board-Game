"""
Sequence Simulator — hyper-realistic Streamlit app with full rules, heuristic AI, batch simulation,
and data logging for probability / ML research.

Requirements:
  pip install streamlit pillow numpy pandas

Save as `sequence_simulator.py` and run:
  streamlit run sequence_simulator.py
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
CARD_W, CARD_H = 72, 100
PADDING = 6
BOARD_BG = (24, 80, 40)
PLAYER_COLORS = [(220, 40, 40), (40, 120, 220), (220, 180, 40), (120, 40, 180),
                 (30, 180, 100), (140, 90, 30), (60,60,60), (180, 60, 100)]
ONE_EYED = {"J♥", "J♠"}
TWO_EYED = {"J♦", "J♣"}
CORNER_POS = {(0,0),(0,9),(9,0),(9,9)}
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]
STANDARD_DECK = [f"{r}{s}" for r in RANKS for s in SUITS]

# -------------------------
# Utilities: Fonts and drawing
# -------------------------
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
    """Draw a single card image with optional chip and highlight."""
    img = Image.new("RGB", (CARD_W, CARD_H), color="white")
    draw = ImageDraw.Draw(img)

    # subtle gradient for realism
    for i in range(CARD_H):
        shade = 255 - int((i/CARD_H)*15)
        draw.line([(0,i),(CARD_W,i)], fill=(shade, shade, shade))

    # border
    border_color = (200,200,200) if not highlight else (255,215,50)
    draw.rectangle([1,1,CARD_W-2,CARD_H-2], outline=border_color, width=2)

    if card=="WILD" or corner_wild:
        draw.rectangle([6,22,CARD_W-6,CARD_H-14], fill=(240,240,210))
        draw.text((CARD_W//2, CARD_H//2-8), "WILD", anchor="mm", font=FONT_MED, fill=(180,30,30))
        draw.ellipse([CARD_W//2-18, CARD_H-36, CARD_W//2+18, CARD_H-12], outline=(200,100,0), width=3)
        if corner_wild:
            draw.rectangle([4,20,CARD_W-4,CARD_H-12], outline=(255,215,0), width=3)
    else:
        rank, suit = card[:-1], card[-1]
        suit_color = (0,0,0) if suit in ["♠","♣"] else (180,0,0)
        # small rank top-left
        draw.text((8,8), rank, font=FONT_SMALL, fill=suit_color)
        draw.text((CARD_W-10, CARD_H-18), suit, font=FONT_SMALL, fill=suit_color, anchor="rm")
        draw.text((CARD_W//2, CARD_H//2-6), suit, anchor="mm", font=FONT_LARGE, fill=suit_color)
        draw.text((8, CARD_H-18), rank, font=FONT_SMALL, fill=suit_color)

    # chip overlay with drop shadow
    if chip_owner is not None and chip_owner > 0:
        color = PLAYER_COLORS[(chip_owner-1) % len(PLAYER_COLORS)]
        cx, cy, r = CARD_W-18, 18, 12
        draw.ellipse([cx-r+2, cy-r+2, cx+r+2, cy+r+2], fill=(0,0,0))  # shadow
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color, outline=(0,0,0))
        draw.text((cx, cy), str(chip_owner), anchor="mm", font=FONT_SMALL, fill=(255,255,255))

    return img

def compose_board_image(board_grid: List[List[str]],
                        chips: List[List[int]],
                        highlights: List[Tuple[int,int]]=[]) -> Image.Image:
    width = COLS*(CARD_W+PADDING) + PADDING
    height = ROWS*(CARD_H+PADDING) + PADDING
    bg = Image.new("RGB", (width, height), color=BOARD_BG)

    y = PADDING
    for r in range(ROWS):
        x = PADDING
        for c in range(COLS):
            card = board_grid[r][c]
            chip_owner = chips[r][c] if chips is not None else None
            highlight = (r,c) in highlights
            corner_wild = (r,c) in CORNER_POS
            card_img = draw_card_image(card, chip_owner=chip_owner, highlight=highlight, corner_wild=corner_wild)
            bg.paste(card_img, (x, y))
            x += CARD_W + PADDING
        y += CARD_H + PADDING
    return bg

# -------------------------
# Board generation
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
                grid[r][c] = tokens[idx]; idx+=1
    return grid

# -------------------------
# Deck & setup
# -------------------------
def create_deck() -> List[str]:
    deck = STANDARD_DECK.copy() + STANDARD_DECK.copy()
    random.shuffle(deck)
    return deck

def get_hand_size(num_players: int) -> int:
    if num_players in [3,4]: return 6
    if num_players == 6: return 5
    if num_players in [8,9]: return 4
    if num_players in [10,12]: return 3
    return 6

# -------------------------
# Sequences detection
# -------------------------
def team_of(player, player_to_team): return player_to_team[player]

def find_sequence_positions(chips, player_to_team, team_id, min_len=5):
    dirs = [(0,1),(1,0),(1,1),(-1,1)]
    seqpos = set()
    def matches(r,c):
        return ((r,c) in CORNER_POS) or (0<=r<ROWS and 0<=c<COLS and chips[r][c]!=0 and player_to_team[chips[r][c]]==team_id)
    for r in range(ROWS):
        for c in range(COLS):
            if not matches(r,c): continue
            for dr,dc in dirs:
                pr,pc = r-dr,c-dc
                if 0<=pr<ROWS and 0<=pc<COLS and matches(pr,pc): continue
                run=[]
                rr,cc = r,c
                while 0<=rr<ROWS and 0<=cc<COLS and matches(rr,cc):
                    run.append((rr,cc)); rr+=dr; cc+=dc
                if len(run)>=min_len: seqpos.update(run)
    return seqpos

def count_team_sequences(chips, player_to_team):
    teams = set(player_to_team.values())
    out = {}
    for t in teams:
        pos = find_sequence_positions(chips, player_to_team, t)
        out[t] = max(0,len(pos)//5)
    return out

# -------------------------
# Legal actions & environment
# -------------------------
Action = Tuple[str, Optional[str], Tuple[int,int]]

def positions_of_card_on_board(board, card):
    return [(r,c) for r in range(ROWS) for c in range(COLS) if board[r][c]==card]

def legal_actions(state, player_id):
    board, chips, hands, completed = state['board'], state['chips'], state['hands'], state['completed']
    hand = hands[player_id]; actions=[]
    for card in hand:
        if card.startswith("J"):
            if card in TWO_EYED:
                for r in range(ROWS):
                    for c in range(COLS):
                        if chips[r][c]==0 and (r,c) not in completed and (r,c) not in CORNER_POS:
                            actions.append(("play_jack_two", card, (r,c)))
            else:
                for r in range(ROWS):
                    for c in range(COLS):
                        if (r,c) in CORNER_POS or (r,c) in completed: continue
                        owner = chips[r][c]
                        if owner!=0 and owner!=player_id: actions.append(("play_jack_one", card, (r,c)))
        else:
            poss = positions_of_card_on_board(board, card)
            placed=False
            for (r,c) in poss:
                if chips[r][c]==0 and (r,c) not in completed:
                    actions.append(("play_card", card, (r,c))); placed=True
            if not placed: actions.append(("turn_in_dead", card, (-1,-1)))
    return actions

def apply_action(state, player_id, action, draw_after=True):
    s = deepcopy(state); typ, card, pos = action
    chips = s['chips']; completed = s['completed']; team_map = s['player_to_team']
    current_team = team_map[player_id]; prev_team_seq = count_team_sequences(chips, team_map).get(current_team,0)

    if typ=="play_card" or typ=="play_jack_two":
        r,c = pos
        s['hands'][player_id].remove(card); s['discard'].append(card)
        chips[r][c] = player_id
    elif typ=="play_jack_one":
        r,c = pos
        s['hands'][player_id].remove(card); s['discard'].append(card)
        chips[r][c] = 0
    elif typ=="turn_in_dead":
        s['hands'][player_id].remove(card); s['discard'].append(card)
    else: raise ValueError("unknown action type")

    if draw_after and s['deck']: s['hands'][player_id].append(s['deck'].pop(0))

    seqpos = find_sequence_positions(chips, team_map, current_team)
    if seqpos: s['completed'].update(seqpos)
    seq_counts = count_team_sequences(chips, team_map); s['seq_counts']=seq_counts
    done = any(v>=(2 if len(set(team_map.values()))==2 else 1) for v in seq_counts.values())
    reward = max(0, seq_counts.get(current_team,0)-prev_team_seq)
    idx = s['players'].index(player_id); s['current_player']=s['players'][(idx+1)%len(s['players'])]
    return s, reward, done, {"action":action}

# -------------------------
# AI: heuristics
# -------------------------
def in_bounds(r,c): return 0<=r<ROWS and 0<=c<COLS

def count_in_direction(chips,r,c,dr,dc,team_map,team_id):
    cnt=0; rr,cc=r+dr,c+dc
    while in_bounds(rr,cc):
        val=chips[rr][cc]
        if (rr,cc) in CORNER_POS or (val!=0 and team_map[val]==team_id):
            cnt+=1; rr+=dr; cc+=dc
        else: break
    return cnt

def evaluate_action_features(state,player_id,action):
    s=state; board,chips,team_map=s['board'],s['chips'],s['player_to_team']; player_team=team_map[player_id]
    typ,card,pos=action; features={f"f{i}":0.0 for i in range(1,11)}
    if typ=="play_jack_one":
        r,c=pos; val=0
        for dr,dc in [(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]:
            if in_bounds(r+dr,c+dc):
                owner=chips[r+dr][c+dc]
                if owner!=0 and team_map[owner]!=player_team: val+=1
        features['f2']=val; features['f5']=val*0.2
        return features
    if typ in ("play_card","play_jack_two"):
        r,c=pos; dirs=[(1,0),(0,1),(1,1),(-1,1)]
        features['f1']=sum((count_in_direction(chips,r,c,dr,dc,team_map,player_team)+count_in_direction(chips,r,c,-dr,-dc,team_map,player_team)>=1) for dr,dc in dirs)/4.0
        # Heuristic 2 simplified
        features['f2']=0.5
        center_r,center_c=(ROWS-1)/2.0,(COLS-1)/2.0
        dist=((r-center_r)**2+(c-center_c)**2)**0.5; maxd=((center_r)**2+(center_c)**2)**0.5
        features['f3']=max(0,1-dist/maxd)
        features['f4']=0.5; features['f5']=-0.1; features['f6']=0.3; features['f7']=0.2
        symr,symc=ROWS-1-r,COLS-1-c; features['f8']=1.0 if in_bounds(symr,symc) and chips[symr][symc]!=0 and team_map[chips[symr][symc]]==player_team else 0.0
        features['f9']=0.3; features['f10']=0.2
    return features

DEFAULT_WEIGHTS = {'f1':2.0,'f2':3.0,'f3':0.8,'f4':1.2,'f5':1.5,'f6':2.2,'f7':1.6,'f8':0.6,'f9':1.0,'f10':1.0}

def score_action(state,player_id,action,weights):
    feats=evaluate_action_features(state,player_id,action)
    score=sum(feats[k]*weights.get(k,1.0) for k in feats)
    typ,card,pos=action
    if pos!=(-1,-1):
        r,c=pos; center_bonus=(ROWS/2-abs(r-ROWS/2)+COLS/2-abs(c-COLS/2))/(ROWS+COLS)
        score+=0.01*center_bonus
    return score

def choose_best_action(state,player_id,weights):
    actions=legal_actions(state,player_id)
    if not actions: return None
    best=None; best_score=-1e9
    for a in actions:
        sc=score_action(state,player_id,a,weights)
        if sc>best_score: best_score=sc; best=a
    return best

# -------------------------
# Simulator
# -------------------------
def new_game(num_players:int, teams:Optional[Dict[int,int]]=None, seed:Optional[int]=None) -> dict:
    if seed is not None: random.seed(seed)
    board = make_official_board(); deck=create_deck()
    hands={p:[deck.pop(0) for _ in range(get_hand_size(num_players))] for p in range(1,num_players+1)}
    chips=[[0]*COLS for _ in range(ROWS)]; completed=set()
    player_to_team = {p:p for p in range(1,num_players+1)} if teams is None else teams
    return {'board':board,'deck':deck,'hands':hands,'chips':chips,'completed':completed,
            'discard':[],'player_to_team':player_to_team,'players':[p for p in range(1,num_players+1)],
            'current_player':1,'seq_counts':{t:0 for t in set(player_to_team.values())}}

def game_step(state,weights):
    player=state['current_player']; action=choose_best_action(state,player,weights)
    if action is None:
        idx=state['players'].index(player); state['current_player']=state['players'][(idx+1)%len(state['players'])]
        return state,{'action':None}
    sc=score_action(state,player,action,weights)
    new_state,reward,done,info=apply_action(state,player,action)
    info.update({'score':sc,'player':player,'reward':reward})
    return new_state,info

def run_batch(n_games,num_players,weights):
    records=[]
    for g in range(n_games):
        s=new_game(num_players); move_count=0
        while True:
            s,info=game_step(s,weights); move_count+=1
            if any(v>=(2 if len(set(s['player_to_team'].values()))==2 else 1) for v in s['seq_counts'].values()):
                winner_team=max(s['seq_counts'].items(),key=lambda kv: kv[1])[0]
                records.append({'game':g,'moves':move_count,'winner_team':winner_team}); break
            if move_count>2000: records.append({'game':g,'moves':move_count,'winner_team':None}); break
    return pd.DataFrame(records)

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(layout="wide", page_title="Sequence Simulator")
st.title("Sequence Simulator — Hyper-Realistic Visuals + AI")

NUM_PLAYERS = st.sidebar.selectbox("Number of Players", [2,3,4,6,8,10,12], index=0)
st.sidebar.markdown("Weights for AI heuristics")
WEIGHTS = {k:st.sidebar.slider(k, -5.0,5.0,DEFAULT_WEIGHTS[k],0.1) for k in DEFAULT_WEIGHTS.keys()}

state = new_game(NUM_PLAYERS)
st.sidebar.text(f"Game Initialized — {NUM_PLAYERS} players")

if st.button("Draw Board"):
    highlights=[]
    board_img = compose_board_image(state['board'], state['chips'], highlights)
    st.image(board_img)

if st.button("Run AI Step"):
    state,info = game_step(state,WEIGHTS)
    highlights = [info['action'][2]] if info.get('action') and info['action'][2] != (-1,-1) else []
    board_img = compose_board_image(state['board'], state['chips'], highlights)
    st.image(board_img)
    st.write(info)

if st.button("Run 10 AI Games"):
    df = run_batch(10, NUM_PLAYERS, WEIGHTS)
    st.dataframe(df)
