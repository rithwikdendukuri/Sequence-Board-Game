# sequence_research.py
# Headless Sequence simulator + analysis (exports CSVs for TI-84)
import argparse, random, os, io, math
from copy import deepcopy
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

# -------------------------
# Game rules (simplified headless) - official rules enforced
# -------------------------
ROWS, COLS = 10, 10
CORNER_POS = {(0,0),(0,9),(9,0),(9,9)}
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
SUITS = ["♠","♥","♦","♣"]
STANDARD_DECK = [f"{r}{s}" for r in RANKS for s in SUITS]
ONE_EYED = {"J♠", "J♥"}
TWO_EYED = {"J♣", "J♦"}

HAND_SIZE_MAP = {2:7,3:6,4:6,6:5,8:4,9:4,10:3,12:3}

def make_official_board(seed=None):
    if seed is not None: random.seed(seed)
    non_jacks = [c for c in STANDARD_DECK if not c.startswith("J")]
    tokens = non_jacks*2
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

def create_deck():
    deck = STANDARD_DECK.copy() + STANDARD_DECK.copy()
    random.shuffle(deck)
    return deck

def get_hand_size(num_players):
    return HAND_SIZE_MAP.get(num_players, 6)

# sequences detection (same as earlier)
def in_bounds(r,c): return 0<=r<ROWS and 0<=c<COLS

def find_sequence_positions(chips, mapping, team_id, min_len=5):
    dirs = [(0,1),(1,0),(1,1),(-1,1)]
    seqpos = set()
    for r in range(ROWS):
        for c in range(COLS):
            if not (((r,c) in CORNER_POS) or (chips[r][c]!=0 and mapping[chips[r][c]]==team_id)):
                continue
            for dr,dc in dirs:
                pr,pc = r-dr, c-dc
                if in_bounds(pr,pc) and (((pr,pc) in CORNER_POS) or (chips[pr][pc]!=0 and mapping[chips[pr][pc]]==team_id)):
                    continue
                run=[]
                rr,cc=r,c
                while in_bounds(rr,cc) and (((rr,cc) in CORNER_POS) or (chips[rr][cc]!=0 and mapping[chips[rr][cc]]==team_id)):
                    run.append((rr,cc))
                    rr+=dr; cc+=dc
                if len(run) >= min_len:
                    seqpos.update(run)
    return seqpos

def count_team_sequences(chips,mapping):
    teams = set(mapping.values())
    out = {}
    for t in teams:
        pos = find_sequence_positions(chips,mapping,t)
        out[t] = len(pos)//5
    return out

# legal actions and apply action
Action = Tuple[str,str,Tuple[int,int]]
def positions_of_card_on_board(board, card):
    return [(r,c) for r in range(ROWS) for c in range(COLS) if board[r][c]==card]

def legal_actions(state, player_id):
    board, chips, hands, completed, mapping = state['board'], state['chips'], state['hands'], state['completed'], state['player_to_team']
    hand = list(hands[player_id])
    actions=[]
    for card in hand:
        if card in TWO_EYED:
            for r in range(ROWS):
                for c in range(COLS):
                    if (r,c) in CORNER_POS or (r,c) in completed: continue
                    if chips[r][c]==0:
                        actions.append(("play_jack_two",card,(r,c)))
        elif card in ONE_EYED:
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
                if chips[r][c]==0 and (r,c) not in completed and (r,c) not in CORNER_POS:
                    actions.append(("play_card", card, (r,c))); placed=True
            if not placed:
                actions.append(("turn_in_dead", card, (-1,-1)))
    return actions

def reshuffle_discard_to_deck(s):
    if s['discard']:
        s['deck'].extend(s['discard'])
        s['discard'].clear()
        random.shuffle(s['deck'])

def draw_card_safe(s):
    if not s['deck'] and s['discard']:
        reshuffle_discard_to_deck(s)
    if not s['deck']:
        return None
    return s['deck'].pop(0)

def apply_action(state, player_id, action, draw_after=True):
    s = deepcopy(state)
    typ, card, pos = action
    chips = s['chips']; completed = s['completed']; mapping = s['player_to_team']
    team_id = mapping[player_id]
    prev_team_seq = count_team_sequences(chips,mapping).get(team_id,0)
    if typ == "play_card":
        r,c = pos
        s['hands'][player_id].remove(card); s['discard'].append(card); chips[r][c]=player_id
    elif typ == "play_jack_two":
        r,c = pos
        s['hands'][player_id].remove(card); s['discard'].append(card); chips[r][c]=player_id
    elif typ == "play_jack_one":
        r,c = pos
        s['hands'][player_id].remove(card); s['discard'].append(card); chips[r][c]=0
    elif typ == "turn_in_dead":
        s['hands'][player_id].remove(card); s['discard'].append(card)
    if draw_after:
        drawn = draw_card_safe(s)
        if drawn: s['hands'][player_id].append(drawn)
    seqpos = find_sequence_positions(chips,mapping,team_id)
    if seqpos: s['completed'].update(seqpos)
    s['seq_counts'] = count_team_sequences(chips,mapping)
    # done?
    num_teams = len(set(mapping.values()))
    required = 2 if num_teams==2 else 1
    done = any(v>=required for v in s['seq_counts'].values())
    new_team_seq = s['seq_counts'].get(team_id,0)
    reward = max(0,new_team_seq-prev_team_seq)
    idx = s['players'].index(player_id)
    s['current_player'] = s['players'][(idx+1)%len(s['players'])]
    return s, reward, done, {"action":action}

# -------------------------
# Strategies (plug-in)
# -------------------------
def random_strategy(state, player_id):
    acts = legal_actions(state, player_id)
    if not acts: return None
    return random.choice(acts)

# Simplified greedy scoring: prefer immediate sequence completion, then block, then center
def greedy_strategy(state, player_id):
    acts = legal_actions(state, player_id)
    if not acts: return None
    best=None; best_score=-1e9
    for a in acts:
        typ,card,pos = a
        score=0.0
        if typ == "turn_in_dead": score -= 1.0
        if typ in ("play_card","play_jack_two"):
            r,c = pos
            # measure immediate sequences created for player's team
            chips = deepcopy(state['chips']); chips2 = deepcopy(chips)
            if typ=="play_card" or typ=="play_jack_two": chips2[r][c]=player_id
            seqs = count_team_sequences(chips2,state['player_to_team']).get(state['player_to_team'][player_id],0)
            score += seqs * 5.0
            # center bonus
            center_r,center_c = (ROWS-1)/2.0,(COLS-1)/2.0
            dist = ((r-center_r)**2 + (c-center_c)**2)**0.5
            score += 1.0 - dist/((center_r**2+center_c**2)**0.5)
        elif typ=="play_jack_one":
            # value removal by counting how many opponent chips adjacent
            r,c = pos; val=0
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if dr==0 and dc==0: continue
                    rr,cc = r+dr, c+dc
                    if in_bounds(rr,cc) and state['chips'][rr][cc]!=0 and state['player_to_team'][state['chips'][rr][cc]]!=state['player_to_team'][player_id]:
                        val+=1
            score += val*2.0
        if score>best_score:
            best_score=score; best=a
    return best

STRATEGIES = {"random": random_strategy, "greedy": greedy_strategy}

# -------------------------
# Simulation runner
# -------------------------
def simulate_single_game(num_players:int, strategy_map:Dict[int,str], seed=None):
    if seed is not None: random.seed(seed)
    board = make_official_board()
    deck = create_deck()
    hands = {p:[deck.pop(0) for _ in range(get_hand_size(num_players))] for p in range(1,num_players+1)}
    chips = [[0]*COLS for _ in range(ROWS)]
    completed = set()
    state = {'board':board,'deck':deck,'hands':hands,'chips':chips,'completed':completed,'discard':[],'player_to_team':{p:p for p in range(1,num_players+1)},'players':[p for p in range(1,num_players+1)],'current_player':1,'seq_counts':{p:0 for p in range(1,num_players+1)}}
    move_count=0
    heat_win = np.zeros(ROWS*COLS, dtype=int)
    winner_team=None; winner_player=None
    while True:
        player = state['current_player']
        strat_name = strategy_map[player]
        strat = STRATEGIES[strat_name]
        action = strat(state, player)
        if action is None:
            # pass
            idx = state['players'].index(player)
            state['current_player'] = state['players'][(idx+1)%len(state['players'])]
            move_count += 1
            if move_count>5000:
                break
            continue
        state, reward, done, info = apply_action(state, player, action)
        move_count += 1
        if done:
            # determine winner team (max seq_counts)
            winner_team = max(state['seq_counts'].items(), key=lambda kv: kv[1])[0]
            # find a player in that team (team labels are player ids identical in this simple mapping)
            winner_player = [p for p,t in state['player_to_team'].items() if t==winner_team][0]
            # mark heat positions from completed set
            for (r,c) in state['completed']:
                heat_win[r*COLS + c] += 1
            break
        if move_count>2000:
            break
    return {
        'moves': move_count,
        'winner_team': winner_team,
        'winner_player': winner_player,
        'seq_counts': state['seq_counts'],
        'heat_win': heat_win
    }

def simulate_n_games(n:int, num_players:int, strat_assign:Dict[int,str], rng_seed_start=0):
    records=[]
    heat_matrix = []
    for i in range(n):
        res = simulate_single_game(num_players, strat_assign, seed=(rng_seed_start+i))
        records.append({'game': i, 'moves': res['moves'], 'winner_team': res['winner_team'], 'winner_player': res['winner_player'], **res['seq_counts']})
        heat_matrix.append(res['heat_win'])
        if (i+1) % max(1, n//10) == 0:
            print(f"Simulated {i+1}/{n} games")
    df = pd.DataFrame(records)
    heat = np.array(heat_matrix)  # n x 100
    return df, heat

# -------------------------
# Statistical analysis functions
# -------------------------
def two_sample_t_test(x, y, equal_var=False):
    # return t-stat, p-val, df, mean_x, mean_y
    tstat, p = stats.ttest_ind(x, y, equal_var=equal_var, nan_policy='omit')
    return {'tstat':float(tstat), 'pval':float(p), 'mean_x':float(np.nanmean(x)), 'mean_y':float(np.nanmean(y))}

def two_proportion_z_test(success_a, n_a, success_b, n_b):
    p1 = success_a / n_a
    p2 = success_b / n_b
    p_pool = (success_a + success_b) / (n_a + n_b)
    se = math.sqrt(p_pool*(1-p_pool)*(1/n_a + 1/n_b))
    z = (p1 - p2) / se if se>0 else 0.0
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return {'z':float(z), 'pval':float(pval), 'p1':p1, 'p2':p2}

def chi2_test_contingency(table):
    # table: 2D array-like
    chi2, p, dof, ex = stats.chi2_contingency(table)
    return {'chi2':float(chi2), 'pval':float(p), 'dof':int(dof), 'expected':ex}

def pca_on_heats(heat_matrix, n_components=5):
    # heat_matrix: n_games x 100 (counts per cell)
    # standardize rows or columns? We'll center columns (cells) then PCA on covariance
    X = heat_matrix.astype(float)
    Xc = X - X.mean(axis=0)
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(Xc)  # scores
    return {'pca':pca, 'scores':comps, 'explained_variance_ratio':pca.explained_variance_ratio_}

# Ledoit-Wolf shrinkage implementation (empirical covariance shrinkage)
def ledoit_wolf_shrinkage(X):
    # X: n_samples x n_features, rows are observations
    # returns shrinkage covariance matrix
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    sample = np.cov(X, rowvar=False, bias=False)
    mu = np.trace(sample) / p
    F = mu * np.eye(p)
    # compute beta (numerator)
    Xc = X - X.mean(axis=0)
    term = (Xc[:,:,None] * Xc[:,None,:])  # n x p x p
    phi_mat = ((term - sample[None,:,:])**2).sum(axis=0) / n
    phi = phi_mat.sum()
    rho = np.sum((sample - F)**2)
    kappa = phi / rho if rho>0 else 0.0
    shrink = max(0, min(1, kappa / n))
    sigma = shrink * F + (1 - shrink) * sample
    return sigma, shrink

# -------------------------
# Export helpers for TI-84
# -------------------------
def export_for_ti84_moves_by_strategy(df, out_path, strategy_names_map):
    """
    Exports a CSV where each column is moves for games where that strategy 'played' as the key player.
    For two-player experiments, strat_assign = {1:'random', 2:'greedy'}, we export columns 'moves_random' and 'moves_greedy'
    For TI-84: import the CSV and each column becomes a List (L1, L2,...)
    """
    # df expected to have: game, moves, winner_team, winner_player, ... plus maybe columns for player ids
    df.to_csv(out_path, index=False)
    print("Saved moves CSV:", out_path)

def export_win_counts_table(df, out_dir, colname='winner_team'):
    counts = df[colname].value_counts().sort_index()
    out = pd.DataFrame({'team':counts.index, 'wins':counts.values})
    out.to_csv(os.path.join(out_dir,'win_counts.csv'), index=False)
    print("Saved win counts:", os.path.join(out_dir,'win_counts.csv'))

# -------------------------
# CLI and orchestration
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_games', type=int, default=2000)
    parser.add_argument('--num_players', type=int, default=2)
    parser.add_argument('--out_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n = args.n_games
    print("Simulating", n, "games...")

    # strategy assignment: player 1 uses random, player 2 uses greedy (2-player example)
    strat_assign = {1:'random', 2:'greedy'}
    df, heat = simulate_n_games(n, args.num_players, strat_assign)
    df.to_csv(os.path.join(args.out_dir,'game_records.csv'), index=False)
    np.save(os.path.join(args.out_dir,'heat_matrix.npy'), heat)
    print("Saved raw outputs.")

    # Basic summaries
    moves_random = df['moves'].values  # here both strategies in same dataframe but games were run with both players
    # For two-player experiments compare statistics conditioned on winner player etc:
    # Example: compare moves distribution when player1 wins vs player2 wins
    p1_wins = df[df['winner_player']==1]['moves'].values
    p2_wins = df[df['winner_player']==2]['moves'].values

    # T-test on moves when P1 wins vs P2 wins
    tres = two_sample_t_test(p1_wins, p2_wins)
    print("T-test (moves P1 wins vs P2 wins):", tres)

    # two-proportion z-test: P1 win rate vs P2 win rate
    n_p1 = (df['winner_player']==1).sum()
    n_p2 = (df['winner_player']==2).sum()
    zres = two_proportion_z_test(n_p1, n, n_p2, n)
    print("two-prop z-test (win rates):", zres)

    # chi2 contingency: starting player (1 vs 2) vs winner
    # build table: rows = starting player (1 or 2), cols = winner (1 or 2)
    df['starting_player'] = 1  # in our simulator current player 1 always starts; extend later
    table = pd.crosstab(df['starting_player'], df['winner_player'])
    chi2res = chi2_test_contingency(table.values)
    print("Chi2 contingency:", chi2res)

    # PCA on heat matrix
    pca_res = pca_on_heats(heat, n_components=6)
    print("PCA explained var ratio:", pca_res['explained_variance_ratio'])

    # Ledoit-Wolf shrinkage on heat covariance
    sigma_hat, shrink = ledoit_wolf_shrinkage(heat.astype(float))
    print("Ledoit-Wolf shrinkage factor:", shrink)

    # Export CSVs for TI-84:
    df.to_csv(os.path.join(args.out_dir,'game_records.csv'), index=False)
    export_win_counts_table(df, args.out_dir)

    # For TI-84 two-sample t-test: export two columns of moves for the two strategies (games where P1 won vs P2 won)
    # We'll create a CSV with two columns 'P1_wins_moves' and 'P2_wins_moves', padded with NaN for unequal lengths
    list1 = p1_wins.tolist()
    list2 = p2_wins.tolist()
    L = max(len(list1), len(list2))
    pad1 = list1 + [np.nan]*(L-len(list1))
    pad2 = list2 + [np.nan]*(L-len(list2))
    pd.DataFrame({'P1_win_moves': pad1, 'P2_win_moves': pad2}).to_csv(os.path.join(args.out_dir,'moves_by_winner_lists.csv'), index=False)
    print("Saved moves_by_winner_lists.csv - import into TI-84 lists for T-Test (two-sample).")

    # For two-proportion z-test (win counts): export a small CSV with [successes, trials] for each group
    pd.DataFrame({'group':['P1','P2'],'wins':[int(n_p1), int(n_p2)], 'trials':[int(n), int(n)]}).to_csv(os.path.join(args.out_dir,'win_counts_for_ztest.csv'), index=False)
    print("Saved win_counts_for_ztest.csv - use values for Z-Test on TI-84 (two-proportion).")

    # For chi-square: export contingency table as CSV
    table_df = pd.DataFrame(table).reset_index()
    table_df.to_csv(os.path.join(args.out_dir,'contingency_table.csv'), index=False)
    print("Saved contingency_table.csv - TI-84 Chi-square test requires observed counts; format depends on calculator version.")

    # Save PCA loadings
    loadings = pca_res['pca'].components_
    pd.DataFrame(loadings).to_csv(os.path.join(args.out_dir,'pca_loadings.csv'), index=False)
    print("All done. Results in", args.out_dir)

if __name__ == "__main__":
    main()
