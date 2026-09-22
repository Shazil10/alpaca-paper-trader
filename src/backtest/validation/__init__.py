"""Validation suite for backtest results.

Four core tests:
1. Parameter stability (heatmap + plateau score)
2. Monte Carlo (block bootstrap, trade shuffle, skip-trade jitter)
3. Cluster analysis (SPP + return-stream clustering)
4. Walk-forward (IS/OOS splits, anchored + rolling, WFE)

Plus overfitting statistics (Deflated Sharpe, PBO via CSCV)
and baselines (random portfolio placebo).
"""
