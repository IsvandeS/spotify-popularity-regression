"""
Q1 plotting script.
Generates heatmap, predicted-vs-actual hexbin, and coefficient bar chart.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# --- Robust path handling ---
base_dir = Path(__file__).parent
csv_path = base_dir / 'spotify_clean.csv'

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

df = pd.read_csv(csv_path)

features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms'
]
features_z = [f + '_z' for f in features]

display = {f: f for f in features}
display['duration_ms'] = 'duration'
display['track_popularity'] = 'popularity'

cols = features + ['track_popularity']
corr = df[cols].corr()

# --- Regression ---
X = df[features_z].values
n = X.shape[0]
A = np.column_stack([np.ones(n), X])
b = df['track_popularity'].values.astype(float)

x_hat = np.linalg.solve(A.T @ A, A.T @ b)
b_hat = A @ x_hat

R2 = 1 - np.sum((b - b_hat)**2) / np.sum((b - b.mean())**2)
rmse = np.sqrt(np.mean((b - b_hat)**2))

# --- Heatmap ---
fig, ax = plt.subplots(figsize=(9, 7.5))
cmap = LinearSegmentedColormap.from_list('rb', ['#2166ac', '#f7f7f7', '#b2182b'])

im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

labels = [display.get(c, c) for c in cols]
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)

for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.values[i, j]
        color = 'white' if abs(v) > 0.5 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', color=color, fontsize=8)

pop_idx = len(cols) - 1
ax.add_patch(plt.Rectangle((-0.5, pop_idx - 0.5), len(cols), 1, fill=False, edgecolor='black', lw=2))
ax.add_patch(plt.Rectangle((pop_idx - 0.5, -0.5), 1, len(cols), fill=False, edgecolor='black', lw=2))

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Pearson correlation', rotation=270, labelpad=15)

ax.set_title(f'Correlation Matrix: Spotify Audio Features + Popularity  (n = {n:,})')

plt.tight_layout()
plt.savefig(base_dir / 'q1_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved q1_heatmap.png')

# --- Predicted vs Actual ---
fig, ax = plt.subplots(figsize=(8, 7))
hb = ax.hexbin(b_hat, b, gridsize=50, cmap='viridis', mincnt=1, bins='log')

cb = plt.colorbar(hb, ax=ax)
cb.set_label('log(count of songs in bin)', rotation=270, labelpad=15)

ax.plot([0, 100], [0, 100], 'r--', lw=1.5)
ax.set_xlim(0, 100)
ax.set_ylim(-2, 102)

ax.set_xlabel('Predicted popularity  (Ax̂)')
ax.set_ylabel('Actual popularity  (b)')
ax.set_title(f'Least-squares fit\nR² = {R2:.4f}, RMSE = {rmse:.1f}, n = {n:,}')

ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(base_dir / 'q1_pred_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved q1_pred_vs_actual.png')

# --- Coefficients ---
coefs = x_hat[1:]
feat_names = [display[f] for f in features]

order = np.argsort(coefs)
ordered_coefs = coefs[order]
ordered_names = [feat_names[i] for i in order]

fig, ax = plt.subplots(figsize=(9, 5.5))

colors = ['#b2182b' if c < 0 else '#2166ac' for c in ordered_coefs]

ax.barh(ordered_names, ordered_coefs, color=colors, edgecolor='black', lw=0.5)
ax.axvline(0, color='black', lw=0.8)

x_max = max(abs(ordered_coefs)) * 1.25
ax.set_xlim(-x_max, x_max)

for i, c in enumerate(ordered_coefs):
    offset = x_max * 0.02
    if c >= 0:
        ax.text(c + offset, i, f'{c:+.2f}', va='center', ha='left')
    else:
        ax.text(c - offset, i, f'{c:+.2f}', va='center', ha='right')

ax.set_xlabel('Coefficient (per +1 SD of feature)')
ax.set_title('Regression coefficients')

ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(base_dir / 'q1_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved q1_coefficients.png')