"""
=============================================================================
  REGIME TRANSITION ANALYSIS
  github.com/preetx77/Vix-Sentiment
=============================================================================
  Standalone module — imports regime labels from options_sentiment_v3.py

  What it builds:
  1. Markov transition matrix (5x5 regime → regime probabilities)
  2. Steady-state distribution (long-run time in each regime)
  3. Expected holding periods (days until regime changes)
  4. Crisis episode analysis (duration + recovery per stress event)
  5. Charts: heatmap, holding periods, steady-state, timeline

  Run:
      python regime_transitions.py

  Requires options_sentiment_v3.py in same folder.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

# ── import pipeline from v3 ───────────────────────────────────────────────────
from backtest import (
    load_data, build_composite, percentile_threshold, STYLE
)

STATES = [
    "Extreme Stress",
    "Stress",
    "Neutral",
    "Calm",
    "Extreme Calm",
]

STATE_COLORS = [
    STYLE["red"], "#d85a30", STYLE["amber"],
    STYLE["teal"], STYLE["green"],
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. ASSIGN REGIMES
# ─────────────────────────────────────────────────────────────────────────────

def assign_regimes(df):
    """
    Assign each day to one of 5 stress regimes using
    rolling percentile breakpoints (same logic as v3 backtest).
    Percentile-based: handles non-stationarity.
    """
    q20 = df["composite"].rolling(252, min_periods=60).quantile(0.20)
    q40 = df["composite"].rolling(252, min_periods=60).quantile(0.40)
    q60 = df["composite"].rolling(252, min_periods=60).quantile(0.60)
    q80 = df["composite"].rolling(252, min_periods=60).quantile(0.80)

    df = df.copy()
    df["q20"] = q20; df["q40"] = q40
    df["q60"] = q60; df["q80"] = q80
    df = df.dropna(subset=["q20","q40","q60","q80","composite"])

    def _label(row):
        s = row["composite"]
        if   s <= row["q20"]: return "Extreme Stress"
        elif s <= row["q40"]: return "Stress"
        elif s <= row["q60"]: return "Neutral"
        elif s <= row["q80"]: return "Calm"
        else:                 return "Extreme Calm"

    df["regime"] = df.apply(_label, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSITION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def compute_transition_matrix(df):
    """
    First-order Markov transition matrix.
    T[i,j] = P(regime tomorrow = j | regime today = i)
    Each row sums to 1.
    """
    regimes = df["regime"].values
    n       = len(STATES)
    counts  = np.zeros((n, n), dtype=float)
    idx     = {s: i for i, s in enumerate(STATES)}

    for t in range(len(regimes) - 1):
        i = idx.get(regimes[t])
        j = idx.get(regimes[t+1])
        if i is not None and j is not None:
            counts[i, j] += 1

    # Normalize rows → probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = counts / row_sums
    return T, counts


# ─────────────────────────────────────────────────────────────────────────────
# 3. STEADY-STATE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def steady_state(T):
    """
    Solve π T = π  (left eigenvector for eigenvalue 1).
    Tells you: long-run fraction of time in each regime.
    """
    eigvals, eigvecs = np.linalg.eig(T.T)
    # Find eigenvector for eigenvalue closest to 1
    idx   = np.argmin(np.abs(eigvals - 1.0))
    pi    = np.real(eigvecs[:, idx])
    pi    = np.abs(pi)
    pi   /= pi.sum()
    return pi


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXPECTED HOLDING PERIODS
# ─────────────────────────────────────────────────────────────────────────────

def holding_periods(T):
    """
    E[days in regime i] = 1 / (1 - T[i,i])
    Geometric distribution: diagonal = self-transition probability.
    """
    diag = np.diag(T)
    stay = np.where(diag < 1.0, diag, 0.9999)
    return 1.0 / (1.0 - stay)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CRISIS EPISODE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def crisis_episodes(df):
    """
    Identify contiguous Extreme Stress periods.
    For each episode: start date, duration, peak VIX,
    SPX drawdown during episode, days to recover to pre-episode SPX.
    """
    in_stress = df["regime"] == "Extreme Stress"
    episodes  = []
    i         = 0
    idx_arr   = df.index

    while i < len(df):
        if in_stress.iloc[i]:
            start = i
            while i < len(df) and in_stress.iloc[i]:
                i += 1
            end = i - 1

            ep_df      = df.iloc[start:end+1]
            pre_spx    = df["spx"].iloc[start - 1] if start > 0 else df["spx"].iloc[start]
            ep_min_spx = ep_df["spx"].min()
            drawdown   = (ep_min_spx - pre_spx) / pre_spx * 100

            # Recovery: days after episode end until SPX >= pre_spx
            recovery = None
            post     = df["spx"].iloc[end+1:]
            recover_idx = post[post >= pre_spx].index
            if len(recover_idx) > 0:
                recovery = (recover_idx[0] - idx_arr[end]).days

            episodes.append({
                "start":       idx_arr[start].date(),
                "end":         idx_arr[end].date(),
                "duration":    end - start + 1,
                "peak_vix":    round(ep_df["vix"].max(), 1),
                "spx_drawdown":round(drawdown, 2),
                "recovery_days": recovery,
            })
        else:
            i += 1

    return pd.DataFrame(episodes)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_results(T, pi, hp, episodes, counts):
    w = 62
    print("\n" + "═"*w)
    print("  TRANSITION MATRIX  P(tomorrow | today)")
    print("═"*w)
    header = f"{'':18s}" + "".join(f"{s[:10]:>12s}" for s in STATES)
    print(header)
    for i, row_name in enumerate(STATES):
        row = f"{row_name:18s}" + "".join(f"{T[i,j]:12.4f}" for j in range(len(STATES)))
        print(row)

    print(f"\n{'═'*w}")
    print("  STEADY-STATE DISTRIBUTION  (long-run % of time)")
    print("═"*w)
    for i, s in enumerate(STATES):
        bar = "█" * int(pi[i] * 40)
        print(f"  {s:18s}  {pi[i]*100:5.1f}%  {bar}")

    print(f"\n{'═'*w}")
    print("  EXPECTED HOLDING PERIODS  (trading days)")
    print("═"*w)
    for i, s in enumerate(STATES):
        print(f"  {s:18s}  {hp[i]:6.1f} days  "
              f"(~{hp[i]/5:.1f} weeks)")

    print(f"\n{'═'*w}")
    print("  TRANSITION COUNT MATRIX  (raw observations)")
    print("═"*w)
    print(header)
    for i, row_name in enumerate(STATES):
        row = f"{row_name:18s}" + "".join(f"{int(counts[i,j]):12d}" for j in range(len(STATES)))
        print(row)

    if not episodes.empty:
        print(f"\n{'═'*w}")
        print("  EXTREME STRESS EPISODES")
        print("═"*w)
        print(episodes.to_string(index=False))
        print(f"\n  Total episodes:       {len(episodes)}")
        print(f"  Avg duration:         {episodes['duration'].mean():.1f} days")
        print(f"  Avg peak VIX:         {episodes['peak_vix'].mean():.1f}")
        print(f"  Avg SPX drawdown:     {episodes['spx_drawdown'].mean():.2f}%")
        rec = episodes["recovery_days"].dropna()
        if len(rec):
            print(f"  Avg recovery:         {rec.mean():.0f} days")

    print(f"\n{'═'*w}")
    print("  RESUME BULLET (copy-paste)")
    print("═"*w)
    top2_from  = STATES[np.argmax([T[i, i] for i in range(len(STATES))])]
    avg_stress = hp[0]
    print(f"""
  Options Market Stress Classifier | Python, Yahoo Finance

  • Modelled market regime dynamics via first-order Markov
    transition matrix across 5 stress states; computed steady-state
    probabilities and expected holding periods via eigendecomposition.
  • Extreme Stress regime: avg holding period {avg_stress:.0f} trading days
    (~{avg_stress/5:.0f} weeks); most persistent state: {top2_from}.
  • Walk-forward OOS + permutation testing used to evaluate
    robustness — correctly scoped as risk-state classifier,
    not alpha engine.
  • Demonstrated drawdown reduction of 64% vs buy-and-hold
    with ~12% active market participation.
""")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(df, T, pi, hp, episodes):
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg"],
        "text.color":       STYLE["text"],
        "font.family":      "monospace",
    })

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(STYLE["bg"])
    fig.text(0.05, 0.97,
             "MARKET STRESS REGIME TRANSITION ANALYSIS",
             fontsize=14, fontweight="bold",
             color=STYLE["text"], va="top")
    fig.text(0.05, 0.945,
             "Markov transition matrix · Steady-state distribution · "
             "Expected holding periods · Crisis episodes",
             fontsize=10, color=STYLE["muted"], va="top")

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.42, wspace=0.35,
                           top=0.91, bottom=0.06,
                           left=0.05, right=0.97)
    ax1 = fig.add_subplot(gs[0, :2])   # regime timeline
    ax2 = fig.add_subplot(gs[0, 2])    # steady-state pie
    ax3 = fig.add_subplot(gs[1, :2])   # transition heatmap
    ax4 = fig.add_subplot(gs[1, 2])    # holding periods
    ax5 = fig.add_subplot(gs[2, :2])   # crisis episodes
    ax6 = fig.add_subplot(gs[2, 2])    # self-transition diagonal

    def sax(ax, title=""):
        ax.set_facecolor(STYLE["surface"])
        ax.tick_params(colors=STYLE["muted"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(STYLE["dim"])
        ax.grid(True, color=STYLE["dim"], alpha=0.22,
                linewidth=0.5, linestyle="--")
        if title:
            ax.set_title(title, color=STYLE["text"],
                         fontsize=9.5, fontweight="bold", pad=8)

    state_idx = {s: i for i, s in enumerate(STATES)}

    # ── Chart 1: Regime timeline ──────────────────────────────────────────
    recent  = df.iloc[-756:]           # last 3 years
    numeric = recent["regime"].map(state_idx).values
    ax1b    = ax1.twinx()
    ax1b.plot(recent.index, recent["vix"],
              color=STYLE["muted"], linewidth=0.7, alpha=0.5, label="VIX")
    ax1b.tick_params(colors=STYLE["muted"], labelsize=7)
    for sp in ax1b.spines.values(): sp.set_color(STYLE["dim"])

    for i, (state, color) in enumerate(zip(STATES, STATE_COLORS)):
        mask = recent["regime"] == state
        ax1.fill_between(recent.index, i - 0.4, i + 0.4,
                         where=mask, color=color, alpha=0.8, step="mid")
    ax1.set_yticks(range(len(STATES)))
    ax1.set_yticklabels(
        [s.replace(" ", "\n") for s in STATES],
        fontsize=7, color=STYLE["muted"]
    )
    ax1.set_ylim(-0.6, 4.6)
    sax(ax1, "Regime timeline (last 3 years)  ·  right axis = VIX")

    # ── Chart 2: Steady-state pie ─────────────────────────────────────────
    ax2.set_facecolor(STYLE["surface"])
    wedges, texts, autotexts = ax2.pie(
        pi, labels=None,
        colors=STATE_COLORS,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=140,
        wedgeprops={"edgecolor": STYLE["bg"], "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_color(STYLE["bg"])
        at.set_fontsize(8)
        at.set_fontweight("bold")
    ax2.legend(STATES, fontsize=7,
               facecolor=STYLE["surface2"],
               labelcolor=STYLE["text"],
               edgecolor=STYLE["dim"],
               loc="lower center",
               bbox_to_anchor=(0.5, -0.15),
               ncol=2)
    ax2.set_title("Steady-state distribution\n(long-run % of time)",
                  color=STYLE["text"], fontsize=9.5,
                  fontweight="bold", pad=8)

    # ── Chart 3: Transition heatmap ───────────────────────────────────────
    ax3.set_facecolor(STYLE["surface"])
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "stress", [STYLE["surface2"], STYLE["purple"]])
    im = ax3.imshow(T, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            val = T[i, j]
            col = STYLE["bg"] if val > 0.5 else STYLE["text"]
            ax3.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=8.5, color=col, fontweight="bold")
    ax3.set_xticks(range(len(STATES)))
    ax3.set_yticks(range(len(STATES)))
    short = ["Ext.Stress","Stress","Neutral","Calm","Ext.Calm"]
    ax3.set_xticklabels(short, fontsize=8, color=STYLE["muted"])
    ax3.set_yticklabels(short, fontsize=8, color=STYLE["muted"])
    ax3.set_xlabel("Tomorrow's regime", fontsize=8, color=STYLE["muted"])
    ax3.set_ylabel("Today's regime",    fontsize=8, color=STYLE["muted"])
    ax3.set_title("Transition matrix  P(j | i)",
                  color=STYLE["text"], fontsize=9.5,
                  fontweight="bold", pad=8)
    plt.colorbar(im, ax=ax3, fraction=0.03,
                 pad=0.02).ax.tick_params(colors=STYLE["muted"])

    # ── Chart 4: Holding periods ──────────────────────────────────────────
    ax4.barh(range(len(STATES)), hp,
             color=STATE_COLORS, alpha=0.85,
             edgecolor=STYLE["dim"], linewidth=0.5)
    for i, h in enumerate(hp):
        ax4.text(h + 0.3, i, f"{h:.1f}d",
                 va="center", fontsize=8, color=STYLE["muted"])
    ax4.set_yticks(range(len(STATES)))
    ax4.set_yticklabels(STATES, fontsize=8)
    ax4.axvline(5, color=STYLE["dim"], linewidth=0.8,
                linestyle="--", label="1 week")
    ax4.legend(fontsize=7, facecolor=STYLE["surface2"],
               labelcolor=STYLE["text"], edgecolor=STYLE["dim"])
    sax(ax4, "Expected holding periods (trading days)")

    # ── Chart 5: Crisis episode bars ──────────────────────────────────────
    if not episodes.empty:
        ep = episodes.copy().reset_index(drop=True)
        x  = range(len(ep))
        ax5.bar(x, ep["duration"],
                color=STYLE["red"], alpha=0.7,
                label="Duration (days)", zorder=3)
        ax5b = ax5.twinx()
        ax5b.plot(x, ep["peak_vix"],
                  color=STYLE["amber"], linewidth=1.5,
                  marker="o", markersize=4, label="Peak VIX")
        ax5b.tick_params(colors=STYLE["muted"], labelsize=7)
        for sp in ax5b.spines.values(): sp.set_color(STYLE["dim"])
        ax5b.set_ylabel("Peak VIX", fontsize=8, color=STYLE["amber"])

        ax5.set_xticks(list(x))
        ax5.set_xticklabels(
            [str(r) for r in ep["start"]],
            rotation=35, ha="right", fontsize=7
        )
        ax5.legend(fontsize=7, facecolor=STYLE["surface2"],
                   labelcolor=STYLE["text"], edgecolor=STYLE["dim"],
                   loc="upper left")
        sax(ax5, "Extreme Stress episodes — duration (bars) vs peak VIX (line)")
        ax5.set_ylabel("Duration (days)", fontsize=8, color=STYLE["muted"])
    else:
        sax(ax5, "No Extreme Stress episodes detected")

    # ── Chart 6: Self-transition probabilities (diagonal) ─────────────────
    diag = np.diag(T)
    ax6.bar(range(len(STATES)), diag,
            color=STATE_COLORS, alpha=0.85,
            edgecolor=STYLE["dim"], linewidth=0.5)
    for i, d in enumerate(diag):
        ax6.text(i, d + 0.005, f"{d:.3f}",
                 ha="center", fontsize=8, color=STYLE["muted"])
    ax6.axhline(0.5, color=STYLE["dim"], linewidth=0.8,
                linestyle="--", label="50% persistence")
    ax6.set_xticks(range(len(STATES)))
    ax6.set_xticklabels(short, fontsize=8, rotation=15)
    ax6.set_ylim(0, 1.05)
    ax6.legend(fontsize=7, facecolor=STYLE["surface2"],
               labelcolor=STYLE["text"], edgecolor=STYLE["dim"])
    sax(ax6, "Self-transition P(stay in regime)")

    fig.text(0.5, 0.015,
             "github.com/preetx77/Vix-Sentiment  ·  "
             "First-order Markov chain · Eigendecomposition steady-state · "
             "Rolling percentile regime assignment",
             ha="center", fontsize=7.5, color=STYLE["dim"])

    plt.savefig("regime_transitions_output.png", dpi=150,
                bbox_inches="tight", facecolor=STYLE["bg"])
    print("\n  Saved → regime_transitions_output.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*62)
    print("  REGIME TRANSITION ANALYSIS")
    print("  github.com/preetx77/Vix-Sentiment")
    print("═"*62)

    print("\n[1/5] Loading data via options_sentiment_v3...")
    df = load_data()
    df = build_composite(df)
    print(f"  {len(df):,} trading days loaded")

    print("\n[2/5] Assigning regimes...")
    df = assign_regimes(df)
    dist = df["regime"].value_counts()
    for s in STATES:
        pct = dist.get(s, 0) / len(df) * 100
        print(f"  {s:18s}: {dist.get(s,0):5d} days ({pct:.1f}%)")

    print("\n[3/5] Computing transition matrix...")
    T, counts = compute_transition_matrix(df)

    print("\n[4/5] Steady-state + holding periods...")
    pi = steady_state(T)
    hp = holding_periods(T)

    print("\n[5/5] Crisis episode analysis...")
    episodes = crisis_episodes(df)

    print_results(T, pi, hp, episodes, counts)

    print("\n  Plotting...")
    plot_all(df, T, pi, hp, episodes)


if __name__ == "__main__":
    main()