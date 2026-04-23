"""
Q1 analysis for Spotify popularity prediction.
- Computes an 11x11 correlation matrix (10 audio features + popularity).
- Fits least squares via normal equations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Robust path handling ---
base_dir = Path(__file__).parent
csv_path = base_dir / 'spotify_clean.csv'

df = pd.read_csv(csv_path)
print(f"Loaded {len(df):,} rows, {df.shape[1]} cols")

features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms'
]
features_z = [f + '_z' for f in features]

cols = features + ['track_popularity']
corr = df[cols].corr()

print("\n=== 11x11 Correlation matrix ===")
print(corr.round(3).to_string())

pop_corr = corr['track_popularity'].drop('track_popularity').sort_values()

print("\n=== Correlations with track_popularity (sorted) ===")
for feat, r in pop_corr.items():
    print(f"  {feat:20s} {r:+.4f}")

print("\nTop 3 NEGATIVE correlations with popularity:")
for feat, r in pop_corr.head(3).items():
    print(f"  {feat:20s} {r:+.4f}")

print("Top 3 POSITIVE correlations with popularity:")
for feat, r in pop_corr.tail(3)[::-1].items():
    print(f"  {feat:20s} {r:+.4f}")

# --- Regression ---
X = df[features_z].values
n = X.shape[0]
A = np.column_stack([np.ones(n), X])
b = df['track_popularity'].values.astype(float)

AtA = A.T @ A
Atb = A.T @ b
x_hat = np.linalg.solve(AtA, Atb)

print("\n=== Least squares setup ===")
print(f"A shape: {A.shape}  (n={n} songs, p={A.shape[1]} cols incl. intercept)")
print(f"A^T A shape: {AtA.shape}")
print(f"cond(A^T A) = {np.linalg.cond(AtA):.2e}")

b_hat = A @ x_hat
resid = b - b_hat
resid_norm = np.linalg.norm(resid)

ss_res = float(np.sum(resid**2))
ss_tot = float(np.sum((b - b.mean())**2))
R2 = 1 - ss_res / ss_tot

print("\n=== Regression coefficients (popularity units per 1 SD of feature) ===")
print(f"  {'intercept':20s} {x_hat[0]:+.4f}")

coefs_sorted = sorted(zip(features, x_hat[1:]), key=lambda t: -abs(t[1]))
for feat, c in coefs_sorted:
    print(f"  {feat:20s} {c:+.4f}")

print("\n=== Fit quality ===")
print(f"Residual norm ||Ax - b|| = {resid_norm:.4f}")
print(f"RMSE                     = {resid_norm/np.sqrt(n):.4f} popularity points")
print(f"R^2                      = {R2:.4f}")
print(f"Mean popularity          = {b.mean():.2f}")
print(f"SD popularity            = {b.std():.2f}")

# --- Save results ---
np.savez(
    base_dir / 'q1_results.npz',
    corr=corr.values,
    corr_cols=np.array(cols),
    coefs=x_hat,
    features=np.array(['intercept'] + features),
    R2=R2,
    resid_norm=resid_norm,
    n=n,
)

print("\nSaved -> q1_results.npz")