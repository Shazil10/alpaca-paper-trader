# Alpaca Paper Trader

Small algo-trading sandbox using Alpaca paper trading.

## Daily run

- Trading run: `.github/workflows/run_bot.yml`
- End-of-day report: `.github/workflows/close_report.yml`

Both workflows write reports into `reports/`:

- `reports/orders_latest.md`
- `reports/orders_latest.html`
- `reports/orders_latest.csv`

## Viewing the close report in VS Code

The close report is generated on GitHub Actions and uploaded as an artifact. To view it locally in VS Code:

1) Install GitHub CLI (gh): https://cli.github.com/
2) Authenticate once:

```bash
gh auth login
```

3) In VS Code, run the task:

- **Terminal → Run Task… → _Reports: Download latest close report_**

This downloads the latest successful close report into:

- `reports/downloaded/`

Then open:

- `reports/downloaded/orders_latest.html` (nice table)
- or `reports/downloaded/orders_latest.md`

### Repo/workflow overrides

If your repo name changes, you can override the repo used by the download script:

```bash
REPO="owner/repo" bash scripts/download_latest_reports.sh
```
