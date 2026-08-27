import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

# --- SYNAPSE plot style -------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'STIXGeneral'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8,
    'axes.linewidth': 0.7,
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'grid.color': '#b0b0b0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'grid.alpha': 0.6,
    'lines.linewidth': 2.0,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '#333333',
    'legend.fancybox': False,
    'figure.autolayout': True,
})

SCORE_COLOR = '#c9c9c9'
MEAN_COLOR = '#21918c'   # viridis teal
BAND_COLOR = '#35b779'   # viridis green


def plot_return(returns, agent, window=100):
    display.clear_output(wait=True)
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_title(f'SYNAPSE : {agent}')

    rolling_mean = pd.Series(returns).rolling(window).mean()
    std = pd.Series(returns).rolling(window).std()

    ax.plot(returns, color=SCORE_COLOR, linewidth=1.0, label='Score')
    ax.plot(rolling_mean, color=MEAN_COLOR, linewidth=2.0, label='Rolling Mean')
    ax.fill_between(range(len(returns)), rolling_mean - std, rolling_mean + std,
                    color=BAND_COLOR, alpha=0.25, linewidth=0)

    # y-range follows the rolling mean band, not the raw score
    lower, upper = (rolling_mean - std).min(), (rolling_mean + std).max()
    if pd.notna(lower) and pd.notna(upper):
        pad = 0.05 * (upper - lower) or 1.0
        ax.set_ylim(lower - pad, upper + pad)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.margins(x=0)
    ax.legend(loc='upper left')
    plt.pause(0.001)
