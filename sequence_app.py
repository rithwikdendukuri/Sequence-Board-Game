import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter

BOARD = 10
SEQ = 5

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(page_title="Sequence Strategy Explorer", layout="wide")

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
.strategy-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
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
# Scoring
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
    scores -= scores.max()          # ← BUG FIX: prevent overflow
    probs = np.exp(scores)
    probs /= probs.sum()
    return empty[np.random.choice(len(empty), p=probs)]

STRATEGIES = {
    "Random":        (random_ai,        "Picks a random empty cell every turn."),
    "Greedy":        (greedy_ai,        "Maximises immediate adjacency to own pieces."),
    "Heuristic":     (heuristic_ai,     "Balances extension, blocking, and clustering."),
    "Probabilistic": (probabilistic_ai, "Samples moves weighted by heuristic softmax."),
}


# ──────────────────────────────────────────────
# Simulation — BUG FIX: track winning cells properly
# ──────────────────────────────────────────────
def simulate_game(A, B):
    board, empty = create_board()
    player = 0
    moves = []
    for _ in range(BOARD*BOARD):
        if not empty:
            return -1, moves, None, None
        move = A(board, empty, player) if player==0 else B(board, empty, player)
        moves.append((move, player))
        place(board, empty, move, player)
        if check_sequence(board, player, move):
            return player, moves, board.copy(), move
        player = 1-player
    return -1, moves, None, None


# ──────────────────────────────────────────────
# Markov / Centrality
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
    """Power iteration — more robust than eigenvector for stochastic matrices."""
    v = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(1000):
        v_new = v @ P
        if np.max(np.abs(v_new - v)) < 1e-9:
            break
        v = v_new
    return v.reshape(BOARD, BOARD)

def eigvals_of(P):
    vals = np.linalg.eigvals(P.T)
    return vals


# ──────────────────────────────────────────────
# Run batch
# ──────────────────────────────────────────────
def run_games(n, A, B, progress_bar=None):
    results, movesets = [], []
    move_counts = np.zeros((BOARD, BOARD))
    win_cell_counts = np.zeros((BOARD, BOARD))   # cells present when win happened
    game_lengths = []

    for i in range(n):
        winner, moves, final_board, winning_move = simulate_game(A, B)
        results.append(winner)
        movesets.append(moves)
        game_lengths.append(len(moves))
        for m, _ in moves:
            move_counts[m] += 1
        if winner != -1 and final_board is not None:
            # mark cells owned by winner
            win_cell_counts += (final_board == winner).astype(float)
        if progress_bar:
            progress_bar.progress((i+1)/n)

    P = build_markov(movesets)
    central = stationary(P)
    eigs = eigvals_of(P)
    seq_prob = win_cell_counts / np.maximum(move_counts, 1)

    return results, move_counts, central, seq_prob, eigs, game_lengths


# ──────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────
CMAP_HEAT   = "magma"
CMAP_COOL   = "viridis"
ACCENT      = "#e8ff6b"

def styled_heatmap(data, title, cmap=CMAP_HEAT, ax=None):
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


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
st.title("⬛ Sequence Strategy Explorer")
st.caption("Simulate head-to-head AI strategies on a 10×10 board · Markov analysis · Eigenvalue spectrum")

col_l, col_r = st.columns([1, 2])

with col_l:
    st.subheader("Configuration")
    nerd = st.toggle("🔬 Nerd Mode", help="Show mathematical details")

    A_choice = st.selectbox("Player A Strategy", list(STRATEGIES.keys()), index=2)
    st.caption(f"_{STRATEGIES[A_choice][1]}_")

    B_choice = st.selectbox("Player B Strategy", list(STRATEGIES.keys()), index=0)
    st.caption(f"_{STRATEGIES[B_choice][1]}_")

    games = st.slider("Number of games", 20, 500, 150, step=10)
    run = st.button("▶ Run Simulation", use_container_width=True, type="primary")

with col_r:
    if not run:
        st.info("Configure strategies on the left and hit **Run Simulation**.")

if run:
    A_fn = STRATEGIES[A_choice][0]
    B_fn = STRATEGIES[B_choice][0]

    pb = st.progress(0, text="Simulating…")
    results, move_counts, central, seq_prob, eigs, game_lengths = run_games(games, A_fn, B_fn, pb)
    pb.empty()

    winsA  = sum(r==0 for r in results)
    winsB  = sum(r==1 for r in results)
    draws  = sum(r==-1 for r in results)
    total  = winsA + winsB   # exclude draws from rate
    rateA  = winsA/total if total else 0
    rateB  = winsB/total if total else 0

    # ── Summary stats ──
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in [
        (c1, "A Wins",  f"{winsA}"),
        (c2, "B Wins",  f"{winsB}"),
        (c3, "Draws",   f"{draws}"),
        (c4, "A Win Rate", f"{rateA:.1%}"),
        (c5, "Avg Game", f"{np.mean(game_lengths):.0f} moves"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
          <div class="stat-value">{val}</div>
          <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Results", "🗺️ Move Heatmap", "🌐 Board Influence",
        "🏆 Win-cell Density", "🔢 Eigenspectrum"
    ])

    # ── Tab 1: Results ──
    with tab1:
        fc1, fc2 = st.columns(2)
        with fc1:
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#0d0d0d")
            ax.set_facecolor("#0d0d0d")
            bars = ax.bar(
                [f"A: {A_choice}", f"B: {B_choice}", "Draw"],
                [winsA, winsB, draws],
                color=["#e8ff6b","#6bbaff","#555"], width=0.5
            )
            ax.set_ylabel("Games won", color="#aaa")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values(): spine.set_edgecolor("#333")
            ax.bar_label(bars, fmt="%d", color="#ddd", padding=3)
            st.pyplot(fig)

        with fc2:
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#0d0d0d")
            ax.set_facecolor("#0d0d0d")
            lengths_counter = Counter(game_lengths)
            xs, ys = zip(*sorted(lengths_counter.items()))
            ax.fill_between(xs, ys, color="#e8ff6b", alpha=0.3)
            ax.plot(xs, ys, color="#e8ff6b", linewidth=1.5)
            ax.set_xlabel("Game length (moves)", color="#aaa")
            ax.set_ylabel("Frequency", color="#aaa")
            ax.tick_params(colors="#aaa")
            for spine in ax.spines.values(): spine.set_edgecolor("#333")
            ax.set_title("Game length distribution", color="#ddd", fontsize=10)
            st.pyplot(fig)

        if nerd:
            st.markdown(f"""<div class="math-block">
Win rate A = {winsA}/{total} = {rateA:.4f}<br>
Win rate B = {winsB}/{total} = {rateB:.4f}<br>
Draw rate  = {draws}/{games} = {draws/games:.4f}<br>
95% CI (A) = [{rateA - 1.96*(rateA*(1-rateA)/total)**0.5:.4f}, {rateA + 1.96*(rateA*(1-rateA)/total)**0.5:.4f}]
</div>""", unsafe_allow_html=True)

    # ── Tab 2: Move Heatmap ──
    with tab2:
        fig = styled_heatmap(move_counts, "Move frequency across all games")
        st.pyplot(fig)
        if nerd:
            st.markdown("""<div class="math-block">
Each cell (i,j) counts how many times it was played across all simulations.<br>
High-frequency cells suggest positional preference of the strategies.
</div>""", unsafe_allow_html=True)

    # ── Tab 3: Board Influence ──
    with tab3:
        fig = styled_heatmap(central, "Stationary distribution (Markov influence)", cmap="plasma")
        st.pyplot(fig)
        if nerd:
            st.markdown("""<div class="math-block">
π = lim_{t→∞} v·P^t  (power iteration, tolerance 1e-9)<br>
π[i] = long-run probability a random walk visits cell i.<br>
Higher values → strategically pivotal board positions.
</div>""", unsafe_allow_html=True)

    # ── Tab 4: Win-cell density ──
    with tab4:
        fig = styled_heatmap(seq_prob, "Win-cell density: P(cell owned | game played)", cmap="inferno")
        st.pyplot(fig)
        if nerd:
            st.markdown("""<div class="math-block">
density[i,j] = (times cell was owned at game-end win) / (times cell was played)<br>
High values = cells that contribute to winning boards.
</div>""", unsafe_allow_html=True)

    # ── Tab 5: Eigenspectrum ──
    with tab5:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#0d0d0d")

        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), color="#333", linewidth=1, linestyle="--")

        re, im = np.real(eigs), np.imag(eigs)
        ax.scatter(re, im, s=8, alpha=0.6, color="#e8ff6b", linewidths=0)

        # highlight dominant eigenvalue
        dominant_idx = np.argmin(np.abs(eigs - 1.0))
        ax.scatter([re[dominant_idx]], [im[dominant_idx]], s=80, color="#ff6b6b",
                   zorder=5, label="λ ≈ 1 (stationary)")

        ax.axhline(0, color="#333", linewidth=0.8)
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("Re(λ)", color="#aaa")
        ax.set_ylabel("Im(λ)", color="#aaa")
        ax.set_title("Eigenvalue spectrum of transition matrix P", color="#ddd", fontsize=10)
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        ax.legend(facecolor="#1a1a1a", labelcolor="#ddd", fontsize=9)
        st.pyplot(fig)

        if nerd:
            second_largest = np.sort(np.abs(eigs))[-2]
            st.markdown(f"""<div class="math-block">
|λ₂| = {second_largest:.6f}  (mixing rate ∝ -1/log|λ₂|)<br>
All eigenvalues of a row-stochastic matrix satisfy |λ| ≤ 1.<br>
Eigenvalues near the unit circle → slow mixing (structured play).<br>
Eigenvalues near 0 → rapid forgetting of positional history.
</div>""", unsafe_allow_html=True)
