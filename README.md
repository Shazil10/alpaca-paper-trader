# Alpaca Paper Trader

Multi-strategy Alpaca paper-trading system with separate strategy modules, lifetime capital caps, order attribution, and scheduled GitHub Actions execution. The trading bot generates current report artifacts after each run so a portfolio microsite can display a fresh paper-trading snapshot instead of hand-maintained numbers.

## Daily run

- Trading run: `.github/workflows/run_bot.yml`
- End-of-day report: `.github/workflows/close_report.yml`

Both workflows write reports into `reports/`:

- `reports/orders_latest.md`
- `reports/orders_latest.html`
- `reports/orders_latest.csv`
- `reports/microsite_snapshot.json`

The close-report workflow commits these report files back to the current branch when they change. A portfolio microsite should read `reports/microsite_snapshot.json` rather than hardcoding strategy counts, capital totals, holdings, or snapshot dates.

## Viewing the close report in VS Code

The close report is generated on GitHub Actions and uploaded as an artifact. To view it locally in VS Code:

1) Install GitHub CLI (gh): https://cli.github.com/
2) Authenticate once:

```bash
gh auth login
```

3) In VS Code, run the task:

- **Terminal → Run Task… → _Reports: Download latest close report_**

Optional: install the extension `Task Buttons` (`spencerwmiles.vscode-task-buttons`).
If installed, you’ll see a clickable **Close Report** button in the VS Code status bar that runs the same task.

This downloads the latest successful close report into:

- `reports/downloaded/`

Then open:

- `reports/downloaded/orders_latest.html` (nice table)
- `reports/downloaded/orders_latest.md`
- `reports/downloaded/microsite_snapshot.json` if you want the latest compact website data payload

### Repo/workflow overrides

If your repo name changes, you can override the repo used by the download script:

```bash
REPO="owner/repo" bash scripts/download_latest_reports.sh
```
