import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from itertools import combinations
import math

BOARD = 10
SEQ = 5

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(page_title="Gomoku Strategy Explorer", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.5px;
}
.stat-card {
    background: #0f0f0f;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 6px 0;
}
.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #e8ff6b;
}
.stat-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.math-block {
    background: #111;
    border-left: 3px solid #e8ff6b;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #ccc;
    border-radius: 0 4px 4px 0;
    margin: 8px 0;
}
.insight-box {
    background: #0a1a0a;
    border: 1px solid #1a3a1a;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #aaddaa;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Board utilities
# ──────────────────────────────────────────────
def create_board():
    return np.full((BOARD, BOARD), -1), [(i, j) for i in range(BOARD) for j in range(BOARD)]

def place(board, empty, move, player):
    board[move] = player
    empty.remove(move)

def check_sequence(board, player, move):
    x, y = move
    for dx, dy in [(1,0),(0,1),(1,1),(1,-1)]:
        count = 1
        for s in range(1, SEQ):
            nx, ny = x+dx*s, y+dy*s
            if 0<=nx<BOARD and 0<=ny<BOARD and board[nx,ny]==player: count+=1
            else: break
        for s in range(1, SEQ):
            nx, ny = x-dx*s, y-dy*s
            if 0<=nx<BOARD and 0<=ny<BOARD and board[nx,ny]==player: count+=1
            else: break
        if count >= SEQ:
            return True
    return False


# ──────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────
def adjacency(board, player, pos):
    x, y = pos
    return sum(
        1 for dx in [-1,0,1] for dy in [-1,0,1]
        if (dx or dy) and 0<=x+dx<BOARD and 0<=y+dy<BOARD and board[x+dx,y+dy]==player
    )

def line_extension(board, player, pos):
    x, y = pos
    score = 0
    for dx, dy in [(1,0),(0,1),(1,1),(1,-1)]:
        for sign in [1, -1]:
            for s in range(1, SEQ):
                nx, ny = x+sign*dx*s, y+sign*dy*s
                if 0<=nx<BOARD and 0<=ny<BOARD and board[nx,ny]==player: score+=1
                else: break
    return score

def heuristic_score(board, player, opponent, pos):
    return 3*line_extension(board,player,pos) + 2*line_extension(board,opponent,pos) + adjacency(board,player,pos)


# ──────────────────────────────────────────────
# Candidate move pruning for minimax
# Only look at cells within radius 2 of existing pieces — huge speedup
# ──────────────────────────────────────────────
def candidate_moves(board, empty):
    if not any(board[i,j] != -1 for i in range(BOARD) for j in range(BOARD)):
        return [(BOARD//2, BOARD//2)]  # open on empty board: center
    candidates = set()
    for i in range(BOARD):
        for j in range(BOARD):
            if board[i,j] != -1:
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni, nj = i+di, j+dj
                        if 0<=ni<BOARD and 0<=nj<BOARD and board[ni,nj]==-1:
                            candidates.add((ni,nj))
    result = [m for m in candidates if m in empty]
    return result if result else empty[:10]  # fallback


# ──────────────────────────────────────────────
# Minimax with alpha-beta pruning
# Depth 2 keeps it fast enough; depth 3 works for late game
# ──────────────────────────────────────────────
def minimax(board, empty, depth, alpha, beta, maximising, player):
    opp = 1 - player
    current = player if maximising else opp

    # Terminal / depth checks
    if depth == 0 or not empty:
        # static eval: heuristic from player's perspective
        score = 0
        for pos in empty:
            score += heuristic_score(board, player, opp, pos) * 0.1
        return score, None

    moves = candidate_moves(board, empty)
    best_move = moves[0]

    if maximising:
        best_val = -math.inf
        for move in moves:
            board[move] = current
            empty.remove(move)
            if check_sequence(board, current, move):
                board[move] = -1
                empty.append(move)
                return 1000, move
            val, _ = minimax(board, empty, depth-1, alpha, beta, False, player)
            board[move] = -1
            empty.append(move)
            if val > best_val:
                best_val, best_move = val, move
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        best_val = math.inf
        for move in moves:
            board[move] = current
            empty.remove(move)
            if check_sequence(board, current, move):
                board[move] = -1
                empty.append(move)
                return -1000, move
            val, _ = minimax(board, empty, depth-1, alpha, beta, True, player)
            board[move] = -1
            empty.append(move)
            if val < best_val:
                best_val, best_move = val, move
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val, best_move

def minimax_ai(board, empty, player):
    _, move = minimax(board.copy(), list(empty), depth=2,
                      alpha=-math.inf, beta=math.inf,
                      maximising=True, player=player)
    return move if move else random.choice(empty)


# ──────────────────────────────────────────────
# AI Strategies
# ──────────────────────────────────────────────
def random_ai(board, empty, player):
    return random.choice(empty)

def greedy_ai(board, empty, player):
    scores = [adjacency(board, player, p) for p in empty]
    m = max(scores)
    return random.choice([empty[i] for i,s in enumerate(scores) if s==m])

def heuristic_ai(board, empty, player):
    opp = 1-player
    scores = [heuristic_score(board,player,opp,p) for p in empty]
    m = max(scores)
    return random.choice([empty[i] for i,s in enumerate(scores) if s==m])

def probabilistic_ai(board, empty, player):
    opp = 1-player
    scores = np.array([heuristic_score(board,player,opp,p) for p in empty], dtype=float)
    scores -= scores.max()
    probs = np.exp(scores)
    probs /= probs.sum()
    return empty[np.random.choice(len(empty), p=probs)]

STRATEGIES = {
    "Random":        (random_ai,        "Picks a random empty cell every turn."),
    "Greedy":        (greedy_ai,        "Maximises immediate adjacency to own pieces."),
    "Heuristic":     (heuristic_ai,     "Balances extension, blocking, and clustering."),
    "Probabilistic": (probabilistic_ai, "Samples moves weighted by heuristic softmax."),
    "Minimax":       (minimax_ai,       "Alpha-beta pruned lookahead, depth 2."),
}


# ──────────────────────────────────────────────
# Simulate one game — returns entropy trace too
# ──────────────────────────────────────────────
def simulate_game(A, B, track_entropy=False):
    board, empty = create_board()
    player = 0
    moves = []
    entropy_trace = []  # Shannon entropy of move distribution at each step

    for _ in range(BOARD*BOARD):
        if not empty:
            return -1, moves, None, entropy_trace

        if track_entropy:
            opp = 1 - player
            fn = A if player == 0 else B
            scores = np.array([heuristic_score(board, player, opp, p) for p in empty], dtype=float)
            if scores.max() > scores.min():
                scores -= scores.max()
                probs = np.exp(scores * 0.5)
                probs /= probs.sum()
                H = -np.sum(probs * np.log(probs + 1e-12))
            else:
                H = np.log(len(empty))
            entropy_trace.append(H)

        move = A(board, empty, player) if player==0 else B(board, empty, player)
        moves.append((move, player))
        place(board, empty, move, player)
        if check_sequence(board, player, move):
            return player, moves, board.copy(), entropy_trace
        player = 1-player

    return -1, moves, None, entropy_trace


# ──────────────────────────────────────────────
# Phase transition: at what move does a game become "decided"?
# We proxy this by tracking when the heuristic lead becomes irreversible
# ──────────────────────────────────────────────
def compute_phase_transition(moves_list, results):
    """
    For each game, find the move number after which the eventual winner
    had a strictly better heuristic sum. Returns an array of "decision move"
    indices — the survival curve is 1 - CDF of this.
    """
    decision_points = []
    for (moves, result) in zip(moves_list, results):
        if result == -1 or len(moves) < SEQ:
            continue
        board = np.full((BOARD, BOARD), -1)
        decided_at = len(moves)  # default: never decided early
        lead_streak = 0
        for idx, (move, player) in enumerate(moves):
            board[move] = player
            # rough balance: count pieces
            p0 = np.sum(board == 0)
            p1 = np.sum(board == 1)
            if result == 0 and p0 > p1 + 3:
                lead_streak += 1
            elif result == 1 and p1 > p0 + 3:
                lead_streak += 1
            else:
                lead_streak = 0
            if lead_streak >= 4:
                decided_at = idx
                break
        decision_points.append(decided_at)
    return np.array(decision_points)


# ──────────────────────────────────────────────
# Markov chain + spectral analysis
# ──────────────────────────────────────────────
def build_markov(move_lists):
    N = BOARD*BOARD
    M = np.zeros((N, N))
    for moves in move_lists:
        cells = [m[0] for m in moves]
        for i in range(len(cells)-1):
            a = cells[i][0]*BOARD+cells[i][1]
            b = cells[i+1][0]*BOARD+cells[i+1][1]
            M[a, b] += 1
    row = M.sum(axis=1, keepdims=True)
    row[row==0] = 1
    return M / row

def stationary(P):
    # Power iteration — numerically stable for row-stochastic matrices
    v = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(2000):
        v_new = v @ P
        if np.max(np.abs(v_new - v)) < 1e-10:
            break
        v = v_new
    return v.reshape(BOARD, BOARD)

def spectral_modes(P, k=3):
    """
    Decompose P^T into eigenvectors. The non-trivial eigenvectors (after the
    stationary one at λ=1) describe 'modes' of board play — which regions
    co-activate. Same idea as PCA but for sequential data.
    """
    vals, vecs = np.linalg.eig(P.T)
    # sort by magnitude descending
    order = np.argsort(-np.abs(vals))
    modes = []
    for i in order[1:k+1]:  # skip index 0 = stationary (λ≈1)
        mode = np.real(vecs[:, i]).reshape(BOARD, BOARD)
        modes.append((np.real(vals[i]), mode))
    return modes, vals


# ──────────────────────────────────────────────
# ELO rating system
# ──────────────────────────────────────────────
def expected_score(ra, rb):
    return 1 / (1 + 10**((rb - ra) / 400))

def update_elo(ra, rb, score_a, K=32):
    ea = expected_score(ra, rb)
    return ra + K * (score_a - ea), rb + K * ((1-score_a) - (1-ea))

def run_round_robin(strategy_names, games_per_pair=40, progress_bar=None):
    """
    Every strategy plays every other strategy `games_per_pair` times.
    Returns elo ratings, full win matrix, and all results per pair.
    """
    fns = {n: STRATEGIES[n][0] for n in strategy_names}
    ratings = {n: 1200.0 for n in strategy_names}
    win_matrix = {n: {m: 0 for m in strategy_names} for n in strategy_names}
    pair_results = {}  # (A,B) -> list of outcomes for bootstrap

    pairs = list(combinations(strategy_names, 2))
    total = len(pairs) * games_per_pair
    done = 0

    for (a, b) in pairs:
        outcomes = []
        for _ in range(games_per_pair):
            winner, moves, board, _ = simulate_game(fns[a], fns[b])
            if winner == 0:
                win_matrix[a][b] += 1
                score_a = 1.0
            elif winner == 1:
                win_matrix[b][a] += 1
                score_a = 0.0
            else:
                score_a = 0.5
            outcomes.append(score_a)
            ratings[a], ratings[b] = update_elo(ratings[a], ratings[b], score_a)
            done += 1
            if progress_bar:
                progress_bar.progress(done / total)
        pair_results[(a,b)] = outcomes

    return ratings, win_matrix, pair_results


# ──────────────────────────────────────────────
# Bootstrap hypothesis test
# H0: strategy A and strategy B win at equal rates
# Returns p-value (two-sided)
# ──────────────────────────────────────────────
def bootstrap_pvalue(outcomes, n_bootstrap=5000):
    outcomes = np.array(outcomes)
    observed = np.mean(outcomes) - 0.5
    null_dist = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(outcomes, size=len(outcomes), replace=True)
        null_dist.append(np.mean(sample) - 0.5)
    null_dist = np.array(null_dist)
    # two-sided: how often does the null exceed |observed|?
    p = np.mean(np.abs(null_dist) >= np.abs(observed))
    return p


# ──────────────────────────────────────────────
# Shannon entropy of a strategy's move choices over a game
# ──────────────────────────────────────────────
def compute_strategy_entropy(strategy_name, n_games=30):
    fn = STRATEGIES[strategy_name][0]
    entropy_by_move = []
    for _ in range(n_games):
        _, _, _, trace = simulate_game(fn, fn, track_entropy=True)
        entropy_by_move.append(trace)
    # pad to same length and average
    max_len = max(len(t) for t in entropy_by_move)
    padded = np.full((n_games, max_len), np.nan)
    for i, t in enumerate(entropy_by_move):
        padded[i, :len(t)] = t
    mean_H = np.nanmean(padded, axis=0)
    return mean_H


# ──────────────────────────────────────────────
# Run the main head-to-head batch
# ──────────────────────────────────────────────
def run_games(n, A, B, progress_bar=None):
    results, movesets = [], []
    move_counts = np.zeros((BOARD, BOARD))
    win_cell_counts = np.zeros((BOARD, BOARD))
    game_lengths = []

    for i in range(n):
        winner, moves, final_board, _ = simulate_game(A, B)
        results.append(winner)
        movesets.append(moves)
        game_lengths.append(len(moves))
        for m, _ in moves:
            move_counts[m] += 1
        if winner != -1 and final_board is not None:
            win_cell_counts += (final_board == winner).astype(float)
        if progress_bar:
            progress_bar.progress((i+1)/n)

    P = build_markov(movesets)
    central = stationary(P)
    modes, eigs = spectral_modes(P, k=3)
    seq_prob = win_cell_counts / np.maximum(move_counts, 1)
    all_eigs = np.linalg.eigvals(P.T)
    decision_pts = compute_phase_transition(movesets, results)

    return results, move_counts, central, seq_prob, all_eigs, game_lengths, modes, decision_pts


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────
def styled_heatmap(data, title, cmap="magma", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    else:
        fig = ax.figure
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=10, color="#ddd", pad=8)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="#555")
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    return fig

def dark_fig(w=5, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    return fig, ax


# ══════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════
st.title("⬛ Gomoku Strategy Explorer")
st.caption("10×10 board · Markov analysis · ELO tournament · Spectral decomposition · Phase transitions")

sidebar_tab, main_tab = st.tabs(["⚙️ Head-to-Head", "🏆 Tournament"])

# ══════════════════════════════════════════════
# HEAD-TO-HEAD TAB
# ══════════════════════════════════════════════
with sidebar_tab:
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("Configuration")
        nerd = st.toggle("🔬 Nerd Mode", help="Show mathematical details")

        A_choice = st.selectbox("Player A Strategy", list(STRATEGIES.keys()), index=2)
        st.caption(f"_{STRATEGIES[A_choice][1]}_")

        B_choice = st.selectbox("Player B Strategy", list(STRATEGIES.keys()), index=0)
        st.caption(f"_{STRATEGIES[B_choice][1]}_")

        games = st.slider("Number of games", 20, 300, 100, step=10)
        if "Minimax" in [A_choice, B_choice]:
            st.warning("⚠️ Minimax is slow — keep games ≤ 60 or be patient.")

        run = st.button("▶ Run Simulation", use_container_width=True, type="primary")

    with col_r:
        if not run:
            st.info("Configure strategies on the left and hit **Run Simulation**.")

    if run:
        A_fn = STRATEGIES[A_choice][0]
        B_fn = STRATEGIES[B_choice][0]

        pb = st.progress(0, text="Simulating games…")
        results, move_counts, central, seq_prob, all_eigs, game_lengths, modes, decision_pts = run_games(
            games, A_fn, B_fn, pb
        )
        pb.empty()

        winsA  = sum(r==0 for r in results)
        winsB  = sum(r==1 for r in results)
        draws  = sum(r==-1 for r in results)
        total  = winsA + winsB
        rateA  = winsA/total if total else 0
        rateB  = winsB/total if total else 0
        ci_hw  = 1.96*(rateA*(1-rateA)/total)**0.5 if total else 0

        # Summary stats row
        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, val in [
            (c1, "A Wins",       f"{winsA}"),
            (c2, "B Wins",       f"{winsB}"),
            (c3, "Draws",        f"{draws}"),
            (c4, "A Win Rate",   f"{rateA:.1%}"),
            (c5, "Avg Game Len", f"{np.mean(game_lengths):.0f}"),
        ]:
            col.markdown(f"""
            <div class="stat-card">
              <div class="stat-value">{val}</div>
              <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📊 Results",
            "🗺️ Move Heatmap",
            "🌐 Markov Influence",
            "🔬 Spectral Modes",
            "🏆 Win-cell Density",
            "📉 Phase Transition",
            "🔢 Eigenspectrum",
        ])

        # ── Tab 1: Results ──────────────────────────────
        with t1:
            fc1, fc2 = st.columns(2)
            with fc1:
                fig, ax = dark_fig(5, 3)
                bars = ax.bar(
                    [f"A: {A_choice}", f"B: {B_choice}", "Draw"],
                    [winsA, winsB, draws],
                    color=["#e8ff6b","#6bbaff","#555"], width=0.5
                )
                ax.set_ylabel("Games won", color="#aaa")
                ax.bar_label(bars, fmt="%d", color="#ddd", padding=3)
                # add 95% CI error bar on A
                ax.errorbar(0, winsA, yerr=ci_hw*total,
                            fmt='none', color='white', capsize=5, linewidth=1.5)
                st.pyplot(fig)

            with fc2:
                fig, ax = dark_fig(5, 3)
                lengths_counter = Counter(game_lengths)
                xs, ys = zip(*sorted(lengths_counter.items()))
                ax.fill_between(xs, ys, color="#e8ff6b", alpha=0.25)
                ax.plot(xs, ys, color="#e8ff6b", linewidth=1.5)
                ax.set_xlabel("Game length (moves)", color="#aaa")
                ax.set_ylabel("Frequency", color="#aaa")
                ax.set_title("Game length distribution", color="#ddd", fontsize=10)
                st.pyplot(fig)

            if nerd:
                st.markdown(f"""<div class="math-block">
Win rate A = {winsA}/{total} = {rateA:.4f}<br>
95% CI (A) = [{rateA - ci_hw:.4f}, {rateA + ci_hw:.4f}]<br>
Draw rate  = {draws}/{games} = {draws/games:.4f}<br>
<br>
CI derived from normal approx: p̂ ± 1.96·√(p̂(1-p̂)/n)
</div>""", unsafe_allow_html=True)

        # ── Tab 2: Move Heatmap ─────────────────────────
        with t2:
            fig = styled_heatmap(move_counts, "Move frequency (all games)", cmap="magma")
            st.pyplot(fig)
            if nerd:
                st.markdown("""<div class="math-block">
f[i,j] = number of times cell (i,j) was played across all games.<br>
Positional clustering → strategy prefers certain board regions.
</div>""", unsafe_allow_html=True)

        # ── Tab 3: Markov stationary ────────────────────
        with t3:
            fig = styled_heatmap(central, "Stationary distribution π (Markov influence)", cmap="plasma")
            st.pyplot(fig)
            if nerd:
                st.markdown("""<div class="math-block">
π = lim_{t→∞} v·Pᵗ  (power iteration, tol=1e-10, max 2000 steps)<br>
π[i] = long-run fraction of turns spent at cell i.<br>
Equivalently: the unique left eigenvector of P at λ=1.
</div>""", unsafe_allow_html=True)

        # ── Tab 4: Spectral Modes ───────────────────────
        with t4:
            st.markdown("""<div class="insight-box">
The Markov transition matrix P can be decomposed into eigenvectors beyond
the stationary distribution. Each subsequent eigenvector describes a "mode"
of play — which board regions activate together. This is analogous to PCA
on positional data, or to normal modes in physics.
</div>""", unsafe_allow_html=True)

            cols = st.columns(len(modes))
            for i, ((eigval, mode), col) in enumerate(zip(modes, cols)):
                with col:
                    fig, ax = plt.subplots(figsize=(3.5, 3.5))
                    fig.patch.set_facecolor("#0d0d0d")
                    ax.set_facecolor("#0d0d0d")
                    im = ax.imshow(mode, cmap="RdBu_r", interpolation="nearest",
                                   norm=plt.Normalize(-np.max(np.abs(mode)), np.max(np.abs(mode))))
                    ax.set_title(f"Mode {i+1}  |λ|={abs(eigval):.3f}", color="#ddd", fontsize=9)
                    ax.tick_params(colors="#555")
                    for spine in ax.spines.values(): spine.set_edgecolor("#222")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)

            if nerd:
                st.markdown("""<div class="math-block">
P = Σᵢ λᵢ vᵢ uᵢᵀ  (spectral decomposition)<br>
Red regions: positively loaded on this mode (move here → mode active)<br>
Blue regions: negatively loaded<br>
|λ| close to 1 → mode persists across many moves (long memory)
</div>""", unsafe_allow_html=True)

        # ── Tab 5: Win-cell density ─────────────────────
        with t5:
            fig = styled_heatmap(seq_prob, "Win-cell density P(owned | game won)", cmap="inferno")
            st.pyplot(fig)
            if nerd:
                st.markdown("""<div class="math-block">
density[i,j] = (times (i,j) was owned by winner at end) / (times (i,j) was played)<br>
Identifies cells that are strategically valuable — not just frequently played.
</div>""", unsafe_allow_html=True)

        # ── Tab 6: Phase Transition ─────────────────────
        with t6:
            st.markdown("""<div class="insight-box">
At what move does a game become "decided"? The survival curve below shows
what fraction of games are still competitive at move k — i.e. the winner
hasn't yet built an irreversible lead. The sharp drop is the phase transition.
</div>""", unsafe_allow_html=True)

            if len(decision_pts) > 0:
                fig, ax = dark_fig(6, 3.5)
                max_move = int(np.max(game_lengths)) + 1
                # survival: fraction of games not yet decided by move k
                survival = np.array([
                    np.mean(decision_pts >= k) for k in range(max_move)
                ])
                xs = np.arange(max_move)
                ax.fill_between(xs, survival, alpha=0.2, color="#e8ff6b")
                ax.plot(xs, survival, color="#e8ff6b", linewidth=2)
                ax.set_xlabel("Move number", color="#aaa")
                ax.set_ylabel("Fraction of games still contested", color="#aaa")
                ax.set_title("Phase transition: when do games become decided?", color="#ddd", fontsize=10)

                # mark median decision point
                median_dp = np.median(decision_pts)
                ax.axvline(median_dp, color="#ff6b6b", linestyle="--", linewidth=1.2,
                           label=f"Median decision: move {median_dp:.0f}")
                ax.legend(facecolor="#1a1a1a", labelcolor="#ddd", fontsize=9)
                st.pyplot(fig)

                if nerd:
                    st.markdown(f"""<div class="math-block">
S(k) = P(game undecided at move k) = 1 - F(k)  [survival function of decision time]<br>
Median decision point: move {median_dp:.1f} of avg {np.mean(game_lengths):.1f}<br>
Games decided in first half: {np.mean(decision_pts < np.mean(game_lengths)/2):.1%}<br>
<br>
A sharp drop in S(k) is a phase transition — a critical point where the
system (the game) loses one macroscopic degree of freedom (contestedness).
</div>""", unsafe_allow_html=True)
            else:
                st.info("Not enough decisive games to plot phase transition. Try more games or stronger strategies.")

        # ── Tab 7: Eigenspectrum ────────────────────────
        with t7:
            fig, ax = dark_fig(5, 5)
            theta = np.linspace(0, 2*np.pi, 300)
            ax.plot(np.cos(theta), np.sin(theta), color="#333", linewidth=1, linestyle="--")

            re, im = np.real(all_eigs), np.imag(all_eigs)
            ax.scatter(re, im, s=8, alpha=0.5, color="#e8ff6b", linewidths=0)

            dominant_idx = np.argmin(np.abs(all_eigs - 1.0))
            ax.scatter([re[dominant_idx]], [im[dominant_idx]], s=80, color="#ff6b6b",
                       zorder=5, label="λ ≈ 1 (stationary)")
            ax.axhline(0, color="#333", linewidth=0.8)
            ax.axvline(0, color="#333", linewidth=0.8)
            ax.set_xlabel("Re(λ)", color="#aaa")
            ax.set_ylabel("Im(λ)", color="#aaa")
            ax.set_title("Eigenvalue spectrum of transition matrix P", color="#ddd", fontsize=10)
            ax.legend(facecolor="#1a1a1a", labelcolor="#ddd", fontsize=9)
            st.pyplot(fig)

            if nerd:
                second = np.sort(np.abs(all_eigs))[-2]
                mixing = -1/np.log(second) if second < 1 and second > 0 else float('inf')
                st.markdown(f"""<div class="math-block">
|λ₂| = {second:.6f}  →  mixing time ≈ {mixing:.1f} steps<br>
All eigenvalues of a row-stochastic matrix: |λ| ≤ 1<br>
λ near unit circle → structured, slow-mixing play (long memory)<br>
λ near 0 → play history is quickly forgotten
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TOURNAMENT TAB
# ══════════════════════════════════════════════
with main_tab:
    st.subheader("Round-Robin ELO Tournament")
    st.caption("Every strategy plays every other strategy N times. ELO ratings + statistical significance testing.")

    t_col1, t_col2 = st.columns([1, 3])
    with t_col1:
        selected = st.multiselect(
            "Strategies to include",
            list(STRATEGIES.keys()),
            default=["Random", "Greedy", "Heuristic", "Probabilistic"]
        )
        gpair = st.slider("Games per matchup", 10, 80, 30, step=10)
        if "Minimax" in selected:
            st.warning("⚠️ Minimax will make this slow. Start with ≤30 games per pair.")
        nerd_t = st.toggle("🔬 Nerd Mode (tournament)", key="nerd_t")
        run_t = st.button("▶ Run Tournament", use_container_width=True, type="primary", key="run_t")

    with t_col2:
        if not run_t:
            st.info("Select strategies and hit **Run Tournament**.")

    if run_t and len(selected) >= 2:
        pb2 = st.progress(0, text="Running round robin…")
        ratings, win_matrix, pair_results = run_round_robin(selected, gpair, pb2)
        pb2.empty()

        # Sort by ELO
        sorted_players = sorted(ratings.keys(), key=lambda x: -ratings[x])

        # ELO Leaderboard
        st.markdown("### ELO Leaderboard")
        elo_cols = st.columns(len(sorted_players))
        colors = ["#e8ff6b", "#aaddff", "#ffaa88", "#ccbbff", "#aaffcc"]
        for i, (name, col) in enumerate(zip(sorted_players, elo_cols)):
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i]
            col.markdown(f"""
            <div class="stat-card">
              <div class="stat-value" style="color:{colors[i]}">{ratings[name]:.0f}</div>
              <div class="stat-label">{medal} {name}</div>
            </div>""", unsafe_allow_html=True)

        # Win matrix heatmap
        st.markdown("### Win Matrix")
        wm_array = np.array([[win_matrix[a][b] for b in sorted_players] for a in sorted_players], dtype=float)
        # normalize by games played per pair
        for i in range(len(sorted_players)):
            for j in range(len(sorted_players)):
                if i != j:
                    total_pair = win_matrix[sorted_players[i]][sorted_players[j]] + win_matrix[sorted_players[j]][sorted_players[i]]
                    if total_pair > 0:
                        wm_array[i,j] /= total_pair

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#0d0d0d")
        im = ax.imshow(wm_array, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(sorted_players)))
        ax.set_yticks(range(len(sorted_players)))
        ax.set_xticklabels(sorted_players, rotation=30, color="#aaa", fontsize=9)
        ax.set_yticklabels(sorted_players, color="#aaa", fontsize=9)
        ax.set_title("Win rate of row vs column (green=row wins more)", color="#ddd", fontsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        for i in range(len(sorted_players)):
            for j in range(len(sorted_players)):
                if i != j:
                    ax.text(j, i, f"{wm_array[i,j]:.2f}", ha='center', va='center',
                            color='black', fontsize=8, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig)

        # Bootstrap significance matrix
        st.markdown("### Statistical Significance (Bootstrap p-values)")
        st.caption("p < 0.05 means we can reject the null hypothesis that the two strategies are equally skilled.")

        pairs = list(combinations(sorted_players, 2))
        sig_matrix = np.full((len(sorted_players), len(sorted_players)), np.nan)
        pval_labels = {}

        for (a, b) in pairs:
            key = (a,b) if (a,b) in pair_results else (b,a)
            outcomes = pair_results[key]
            if key == (b,a):
                outcomes = [1-x for x in outcomes]
            pval = bootstrap_pvalue(outcomes)
            ia = sorted_players.index(a)
            ib = sorted_players.index(b)
            sig_matrix[ia, ib] = pval
            sig_matrix[ib, ia] = pval
            pval_labels[(a,b)] = pval

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        fig2.patch.set_facecolor("#0d0d0d")
        ax2.set_facecolor("#0d0d0d")
        im2 = ax2.imshow(sig_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.2, interpolation="nearest")
        ax2.set_xticks(range(len(sorted_players)))
        ax2.set_yticks(range(len(sorted_players)))
        ax2.set_xticklabels(sorted_players, rotation=30, color="#aaa", fontsize=9)
        ax2.set_yticklabels(sorted_players, color="#aaa", fontsize=9)
        ax2.set_title("p-value (red = significant difference, p<0.05)", color="#ddd", fontsize=9)
        for spine in ax2.spines.values(): spine.set_edgecolor("#333")
        for i in range(len(sorted_players)):
            for j in range(len(sorted_players)):
                if not np.isnan(sig_matrix[i,j]):
                    label = f"{sig_matrix[i,j]:.3f}"
                    color = "white" if sig_matrix[i,j] < 0.05 else "#aaa"
                    ax2.text(j, i, label, ha='center', va='center', color=color, fontsize=8)
        fig2.colorbar(im2, ax=ax2, fraction=0.046)
        st.pyplot(fig2)

        if nerd_t:
            st.markdown("""<div class="math-block">
Bootstrap test: resample game outcomes 5000 times with replacement.<br>
p = P(|bootstrapped mean - 0.5| ≥ |observed mean - 0.5|)<br>
H₀: both strategies win at equal rates (expected mean = 0.5)<br>
Small p → H₀ rejected → one strategy is genuinely better.
</div>""", unsafe_allow_html=True)

        # Shannon entropy comparison
        st.markdown("### Strategy Entropy: How Deterministic Is Each Strategy?")
        st.caption("Lower entropy = more deterministic play. Measured over move choices mid-game.")

        fig3, ax3 = dark_fig(7, 3.5)
        palette = ["#e8ff6b","#6bbaff","#ff6b8a","#aaffcc","#ffcc88"]
        for i, name in enumerate(selected[:5]):
            H_trace = compute_strategy_entropy(name, n_games=20)
            xs = np.arange(len(H_trace))
            ax3.plot(xs, H_trace, label=name, color=palette[i], linewidth=1.5, alpha=0.9)

        ax3.set_xlabel("Move number", color="#aaa")
        ax3.set_ylabel("Shannon entropy H (nats)", color="#aaa")
        ax3.set_title("Move-choice entropy over game progression", color="#ddd", fontsize=10)
        ax3.legend(facecolor="#1a1a1a", labelcolor="#ddd", fontsize=9)
        st.pyplot(fig3)

        if nerd_t:
            st.markdown("""<div class="math-block">
H(X) = -Σ p(x)·log p(x)  [Shannon entropy of softmax move distribution]<br>
Random AI: H = log(|empty|) ≈ max entropy (uniform distribution)<br>
Deterministic AI: H → 0 as it concentrates on one move<br>
Entropy decreasing over game = strategy becoming more certain as board fills.
</div>""", unsafe_allow_html=True)
