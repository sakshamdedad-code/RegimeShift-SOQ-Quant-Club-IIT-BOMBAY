# Regime-Shift Asset Allocation

A regime-aware tactical asset allocation system: detect the market's hidden state (Bull /
Bear / Crisis) with a Hidden Markov Model, then pick equity/gold/bond portfolio weights
for that regime with convex optimization (`cvxpy`) - validated honestly with walk-forward
testing so the backtest isn't secretly cheating by peeking into the future.

## What's in this repo

- `Regime_Shift_Asset_Allocation.ipynb` — the full pipeline, top to bottom, with
  explanations at every phase: data → features → regime detection → walk-forward
  validation → portfolio optimization → backtest → results.
- `README.md` - this file.

## How to run it

```bash
pip install yfinance hmmlearn cvxpy scikit-learn matplotlib pandas numpy scipy nbformat
jupyter notebook Regime_Shift_Asset_Allocation.ipynb
```

Run all cells top to bottom. **You need internet access to Yahoo Finance** for the real
results. If Yahoo Finance isn't reachable (rate-limited, blocked network, sandboxed
environment), the notebook automatically falls back to a synthetic regime-switching market
it generates itself - the pipeline still runs end to end and every output is real, but it's
demonstrating the mechanics rather than real market behavior. The active data source is
always printed clearly at the top of Phase 1, e.g. `Active data source: yfinance (live)` or
`Active data source: synthetic (fallback)`.

## Data

| Asset | Ticker | Source |
|---|---|---|
| Equity | `^NSEI` (Nifty 50) | Yahoo Finance |
| Gold | `GOLDBEES.NS` | Yahoo Finance |
| Bonds | `LTGILTBEES.NS` | Yahoo Finance |
| Volatility | `^INDIAVIX` | Yahoo Finance |

## Key decisions and why

**Why 3 regimes (Bull / Bear / Crisis)?**
Matches the problem statement directly, and is the minimum needed to distinguish "calm,"
"declining," and "acutely stressed" markets — each of which plausibly wants a different
portfolio. More states would let the model chase noise; fewer would collapse genuinely
different market conditions together.

**Why these features (momentum + volatility + VIX + cross-asset gold vol)?**
Momentum captures trend direction, volatility captures how turbulent recent moves have
been, and VIX is the market's own forward-looking fear gauge. Gold's own realized
volatility is included because a genuine flight-to-safety crisis often shows up there too,
not just in equities.

**Why log-transform the volatility-type features before the HMM?**
`vol_1m`, `gold_vol_1m`, and `vix` are strictly positive and right-skewed. A `GaussianHMM`
assumes each state's features are (roughly) Gaussian; log-transforming volatility makes
that assumption much more reasonable and noticeably improves how cleanly the model
separates regimes.

**Why multi-restart HMM fitting?**
EM (the algorithm hmmlearn uses to fit the HMM) only guarantees a *local* optimum. With one
random initialization, it can converge to a degenerate solution - e.g. splitting the
abundant "calm" days into two near-duplicate states while merging the rarer Bear and Crisis
days into a single state, simply because that solution has higher likelihood when one true
regime (Crisis) is much rarer than the others. The fix used here: fit the HMM several times
from different random seeds and keep whichever converged fit has the highest log-likelihood
on the training data — the HMM equivalent of k-means++ restarts.

**Why regime-conditional mu/Sigma in the optimizer, instead of one trailing-window
estimate used for every regime?**
This was the single most important design choice in the whole project. If the optimizer's
expected-return and covariance inputs are the same regardless of which regime is currently
detected, then the risk-aversion parameter (`gamma`) alone can't turn a negative
trailing-window return estimate into an equity-heavy Bull-regime portfolio - the regime
label would barely influence the final allocation. Instead, at every rebalance, expected
returns and covariance are estimated **separately for each regime**, using only
historically-labeled days of that regime seen so far. Because Crisis days are rare
(especially early in the backtest), these regime-conditional estimates are shrunk toward
the unconditional (all-history) estimate using `n / (n + k)` shrinkage, so a handful of
extreme days early on can't produce a wildly unstable estimate.

**Why rebalance on regime-change-or-staleness, not every day?**
Daily rebalancing driven by noisy day-to-day regime flips would generate needless turnover
and let transaction costs quietly eat the strategy's entire edge — this is explicitly
called out as a trap in the project brief. Weights only get a fresh optimization (and pay a
transaction cost) when the detected regime changes, or when `max_hold` trading days have
passed since the last rebalance (a staleness cap so a long-lived regime still gets its
estimates refreshed periodically). Between rebalances, weights drift naturally with asset
returns.

## Avoiding lookahead bias — the checklist this notebook actually follows

1. All engineered features use only backward-looking rolling windows or same-day
   observable levels — never a future value.
2. The walk-forward harness uses an **expanding training window**: at each step, everything
   from the very start up to "now," refit from scratch every step.
3. **Both** the feature scaler (`StandardScaler`, mean/std) **and** the HMM (means,
   covariances, transition matrix) are refit on **train-only** data at every step — fitting
   the scaler on the full dataset before splitting is a classic, easy-to-miss leak.
4. The model only ever labels the **held-out test block** using parameters fit on strictly
   earlier data.
5. The portfolio optimizer's regime-conditional mu/Sigma at each rebalance date are
   estimated using only **strictly-past**, already-walk-forward-labeled days.
6. The exploratory full-sample HMM fit in Phase 3 is clearly marked as **not** the model
   used in the backtest — it exists purely as a first visual sanity check.

## Results

See the notebook's "Results" section for the full performance table (Sharpe, Sortino, max
drawdown, Calmar, turnover) comparing the dynamic regime-shift strategy against a static
60/40 portfolio and an equal-weight portfolio, both with and without transaction costs.
Numbers will differ between a live-data run and the synthetic-fallback run - re-run with
internet access to Yahoo Finance to reproduce the real submission results.

## Reproducing results

The notebook is deterministic given the same data: `np.random.seed(7)` is set at the top,
and the HMM's multi-restart fitting uses fixed, offset-per-window random seeds
(`random_state=100 + step` in the walk-forward loop) so re-running produces the same
regime labels and backtest for the same input data.

## Tech stack

Python 3.9+, `hmmlearn`, `cvxpy`, `NumPy`, `Pandas`, `Matplotlib`, `yfinance`, `scikit-learn`, `SciPy`.
