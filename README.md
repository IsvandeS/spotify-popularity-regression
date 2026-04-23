# Spotify Popularity — Least-Squares Regression

Predict Spotify `track_popularity` (0–100) from 10 standardized audio features, using the normal equations.

## Data

`spotify_clean.csv` — 28,352 tracks.

- **Target:** `track_popularity`
- **Features (z-scored):** danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms

Standardization means coefficients are in **popularity points per +1 SD of feature**.

## Scripts

| File | Purpose |
|---|---|
| `q1_analysis.py` | Builds the 11×11 correlation matrix, solves `AᵀA x̂ = Aᵀb`, reports R² / RMSE / residual norm, saves `q1_results.npz`. |
| `q1_plots.py` | Regenerates the three figures. |

## Usage

```bash
python q1_analysis.py    # prints correlations + coefficients, saves q1_results.npz
python q1_plots.py       # writes the three PNGs
```

Both scripts resolve paths relative to their own location, so they work from any working directory as long as `spotify_clean.csv` sits next to them.

## Results

**Fit quality (n = 28,352)**

| Metric | Value |
|---|---|
| R² | 0.058 |
| RMSE | 23.0 popularity points |
| Mean popularity | ~42 |

**Coefficients, sorted by magnitude**

| Feature | Coef | Feature | Coef |
|---|---:|---|---:|
| energy | −4.25 | danceability | +0.52 |
| loudness | +3.50 | valence | +0.42 |
| duration_ms | −2.65 | speechiness | −0.67 |
| instrumentalness | −2.17 | liveness | −0.67 |
| acousticness | +0.96 | tempo | +0.70 |

## Figures

- `q1_heatmap.png` — 11×11 Pearson correlation matrix, popularity row/column boxed.
- `q1_pred_vs_actual.png` — hexbin of predicted vs. actual popularity with `y = x` reference.
- `q1_coefficients.png` — horizontal bar chart of the 10 coefficients, sorted.

## Interpretation

- **Low R² (≈ 0.06):** audio features alone explain very little of the variance in popularity. Other factors such as genre, artist, marketing, and release timing are not explained by this dataset.
- **Energy vs. loudness:** these correlate at 0.68, and their fitted coefficients (−4.25 and +3.50) are large and opposite in sign. This is a multicollinearity artifact; neither variable correlates strongly with popularity on its own (−0.10 and +0.04 respectively).
- **Predicted-vs-actual plot:** predictions compress into a narrow band around the mean (~20–55), while actuals span the full 0–100 range. This is the visual signature of a weak linear fit on a noisy target.

## Output

`q1_results.npz` contains: `corr`, `corr_cols`, `coefs`, `features`, `R2`, `resid_norm`, `n`.
