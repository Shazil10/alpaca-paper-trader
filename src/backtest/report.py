"""A single self-contained HTML page comparing anything against anything.

Every run already writes its numbers to disk. What was missing is the comparison,
and a comparison is where most of the information is: a Sharpe of 0.8 means
nothing until it sits beside SPY, beside an equal-weight basket of the same names,
beside the same strategy's previous version, and beside the same strategy run with
the costs turned up.

Eight comparisons, each a section:

1. **Headline** -- every stream side by side on the same metrics.
2. **Equity and drawdown** -- normalized to 1.0 at the common start, so books of
   different size are comparable.
3. **Against the benchmark** -- beta, Jensen alpha, information ratio, capture.
4. **Against equal weight** -- isolates selection skill from the universe's drift.
   If equal weight matches the strategy, the ranking adds nothing.
5. **In-sample versus out-of-sample** -- the same metrics on both halves.
6. **Cost scenarios** -- from a research run's cost stress.
7. **Parameters** -- the sweep, with the chosen cell marked.
8. **Regimes and calendar years** -- where the return actually came from.
9. **Previous version** -- and whether the *config* also changed, because two runs
   with different cost or date assumptions are not a strategy comparison.
10. **Correlation** -- with every other stream, over the overlapping window only,
    with the length of that window shown. A correlation over thirty shared days is
    noise wearing a number.

Deliberately no matplotlib and no JavaScript. Charts are inline SVG generated
here, so the output is one file that opens anywhere, survives being emailed, and
cannot drift with a plotting library version. The cost is that the charts are
static and simple, which for an equity curve is all that is needed.
"""

from __future__ import annotations

import html
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest import metrics

logger = logging.getLogger(__name__)

#: Correlation over fewer overlapping sessions than this is reported but flagged.
#: Roughly a quarter: below it, a single week dominates the estimate.
MIN_CORRELATION_SESSIONS = 60

#: Chart geometry. One place, because every section shares it.
CHART_WIDTH = 1000
CHART_HEIGHT = 260

PALETTE = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd",
    "#ff7f0e", "#8c564b", "#17becf", "#7f7f7f",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    """One backtest run, loaded off disk."""
    name: str
    path: Optional[Path] = None
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    stats: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def strategy_id(self) -> str:
        return str(self.config.get("strategy_id", self.name))

    @property
    def window(self) -> str:
        if len(self.equity) == 0:
            return "—"
        return f"{self.equity.index[0]:%Y-%m-%d} .. {self.equity.index[-1]:%Y-%m-%d}"


def load_run(path: Path, name: Optional[str] = None) -> RunData:
    """Read a ``runs/<id>/`` directory into a RunData.

    Metrics are recomputed from the equity curve rather than trusted from
    metrics.json. The stored file was written by whatever version of the code ran
    at the time, and a report that mixes metric definitions across runs is worse
    than one that is slightly slower.
    """
    path = Path(path)
    data = RunData(name=name or path.name, path=path)

    equity_path = path / "equity.csv"
    if equity_path.exists():
        frame = pd.read_csv(equity_path)
        date_col = frame.columns[0]
        value_col = "equity" if "equity" in frame.columns else frame.columns[-1]
        series = pd.Series(
            frame[value_col].to_numpy(dtype=float),
            index=pd.to_datetime(frame[date_col]),
            name=data.name,
        ).dropna()
        data.equity = series
        data.returns = series.pct_change().dropna()

    for filename, attribute in (("config.json", "config"), ("metrics.json", "stats")):
        candidate = path / filename
        if candidate.exists():
            try:
                setattr(data, attribute, json.loads(candidate.read_text()))
            except Exception:
                logger.warning("could not parse %s", candidate)

    return data


def discover_runs(
    runs_dir: Path,
    strategy_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Path]:
    """Run directories, newest last. Research directories are skipped.

    Directory names begin with the run date, so lexical order is chronological --
    which is the only ordering available, since a run does not record which run
    preceded it.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []

    candidates = [
        d for d in sorted(runs_dir.iterdir())
        if d.is_dir() and (d / "equity.csv").exists() and "_research_" not in d.name
    ]

    if strategy_id:
        candidates = [
            d for d in candidates
            if load_run(d).strategy_id == strategy_id
        ]

    return candidates[-limit:] if limit else candidates


# ---------------------------------------------------------------------------
# Metric assembly
# ---------------------------------------------------------------------------

def stream_metrics(
    equity: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """The metric row for one stream. Recomputed, never read from disk."""
    if len(equity) < 3:
        return {}

    returns = equity.pct_change().dropna()
    row: Dict[str, Any] = {
        "CAGR": metrics.cagr(equity),
        "Sharpe": metrics.sharpe_ratio(returns),
        "Sortino": metrics.sortino_ratio(returns),
        "Vol": metrics.annualized_volatility(returns),
        "Max DD": metrics.max_drawdown(equity),
        "Calmar": metrics.calmar_ratio(equity),
        "Win rate": metrics.win_rate(returns),
        "Sessions": len(returns),
    }

    if benchmark_returns is not None and len(benchmark_returns) > 30:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 30:
            port, bench = aligned.iloc[:, 0], aligned.iloc[:, 1]
            row["Beta"] = metrics.beta(port, bench)
            row["Alpha"] = metrics.alpha_jensen(port, bench)
            row["Info ratio"] = metrics.information_ratio(port, bench)

    return row


def _normalize(series: pd.Series, start: Optional[pd.Timestamp] = None) -> pd.Series:
    """Rebase to 1.0 so streams with different capital are comparable."""
    trimmed = series if start is None else series.loc[series.index >= start]
    trimmed = trimmed.dropna()
    if len(trimmed) == 0 or trimmed.iloc[0] == 0:
        return trimmed
    return trimmed / trimmed.iloc[0]


# ---------------------------------------------------------------------------
# SVG charting
# ---------------------------------------------------------------------------

def _polyline(
    series: pd.Series,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    colour: str,
    log_scale: bool = False,
) -> str:
    if len(series) < 2:
        return ""

    xs = series.index.map(pd.Timestamp.timestamp).to_numpy(dtype=float)
    ys = series.to_numpy(dtype=float)

    if log_scale:
        ys = np.log(np.clip(ys, 1e-9, None))

    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    points = []
    for x, y in zip(xs, ys):
        px = (x - x_min) / x_span * CHART_WIDTH
        py = CHART_HEIGHT - (y - y_min) / y_span * CHART_HEIGHT
        points.append(f"{px:.1f},{py:.1f}")

    return (
        f'<polyline fill="none" stroke="{colour}" stroke-width="1.6" '
        f'points="{" ".join(points)}"/>'
    )


def line_chart(
    streams: Dict[str, pd.Series],
    *,
    log_scale: bool = False,
    zero_line: bool = False,
) -> str:
    """Inline SVG for a set of aligned series. No axes labels beyond the extremes.

    Kept intentionally plain: the purpose is shape and relative position, and a
    reader who needs exact values has the tables above.
    """
    usable = {k: v.dropna() for k, v in streams.items() if len(v.dropna()) > 1}
    if not usable:
        return "<p class='muted'>nothing to plot</p>"

    x_values = np.concatenate([
        s.index.map(pd.Timestamp.timestamp).to_numpy(dtype=float)
        for s in usable.values()
    ])
    y_values = np.concatenate([s.to_numpy(dtype=float) for s in usable.values()])
    if log_scale:
        y_values = np.log(np.clip(y_values, 1e-9, None))

    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    pad = (y_max - y_min) * 0.05 or 0.05
    y_min, y_max = y_min - pad, y_max + pad

    parts = [
        f'<svg class="chart" viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" '
        f'preserveAspectRatio="none" role="img">'
    ]

    if zero_line and y_min <= 0 <= y_max:
        zero_y = CHART_HEIGHT - (0 - y_min) / (y_max - y_min) * CHART_HEIGHT
        parts.append(
            f'<line x1="0" y1="{zero_y:.1f}" x2="{CHART_WIDTH}" y2="{zero_y:.1f}" '
            f'stroke="#bbb" stroke-width="1" stroke-dasharray="4 4"/>'
        )

    for index, (name, series) in enumerate(usable.items()):
        parts.append(_polyline(
            series, x_min, x_max, y_min, y_max,
            PALETTE[index % len(PALETTE)], log_scale,
        ))

    parts.append("</svg>")

    legend = " ".join(
        f'<span class="key"><i style="background:{PALETTE[i % len(PALETTE)]}"></i>'
        f"{html.escape(name)}</span>"
        for i, name in enumerate(usable)
    )
    first = min(s.index[0] for s in usable.values())
    last = max(s.index[-1] for s in usable.values())

    return (
        "".join(parts)
        + f'<div class="legend">{legend}</div>'
        + f'<div class="muted">{first:%Y-%m-%d} to {last:%Y-%m-%d}'
        + (" · log scale" if log_scale else "")
        + "</div>"
    )


# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.3f}"
    return html.escape(str(value))


PERCENT_ROWS = ("CAGR", "Vol", "Max DD", "Alpha", "Win rate")


def _is_percent(key: str) -> bool:
    """Prefix match, so derived rows like "CAGR − SPY" format as percentages too."""
    return key.startswith(PERCENT_ROWS)


def metric_table(rows: Dict[str, Dict[str, Any]]) -> str:
    """Streams as columns, metrics as rows -- the orientation that reads best.

    Comparison is across streams, so putting them side by side keeps the numbers
    being compared adjacent instead of a screen apart.
    """
    if not rows:
        return "<p class='muted'>no data</p>"

    names = list(rows)
    keys: List[str] = []
    for row in rows.values():
        for key in row:
            if key not in keys:
                keys.append(key)

    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    body = []
    for key in keys:
        cells = []
        for name in names:
            value = rows[name].get(key)
            if _is_percent(key) and isinstance(value, (float, np.floating)):
                cells.append(f"<td>{value:.1%}</td>")
            else:
                cells.append(f"<td>{_fmt(value)}</td>")
        body.append(f"<tr><th class='rowkey'>{html.escape(key)}</th>{''.join(cells)}</tr>")

    return (
        f"<table><thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def frame_table(frame: pd.DataFrame, highlight: Optional[int] = None) -> str:
    """A DataFrame as HTML, with an optional highlighted row index."""
    if frame is None or len(frame) == 0:
        return "<p class='muted'>no data</p>"

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = []
    for position, (_, row) in enumerate(frame.iterrows()):
        cells = "".join(f"<td>{_fmt(v)}</td>" for v in row)
        css = " class='mark'" if highlight is not None and position == highlight else ""
        body.append(f"<tr{css}>{cells}</tr>")

    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def section(title: str, body: str, note: str = "") -> str:
    note_html = f"<p class='note'>{html.escape(note)}</p>" if note else ""
    return f"<section><h2>{html.escape(title)}</h2>{note_html}{body}</section>"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def correlation_section(streams: Dict[str, pd.Series]) -> str:
    """Pairwise correlation of daily returns over the overlapping window only.

    The window length is shown because it governs whether the number means
    anything. Two strategies whose backtests share a single quarter can report a
    correlation of 0.9 or -0.9 on the strength of one bad week.
    """
    names = [k for k, v in streams.items() if len(v.dropna()) > 5]
    if len(names) < 2:
        return "<p class='muted'>need two streams to correlate</p>"

    returns = {n: streams[n].pct_change().dropna() for n in names}

    corr = pd.DataFrame(index=names, columns=names, dtype=float)
    overlap = pd.DataFrame(index=names, columns=names, dtype="Int64")

    for a in names:
        for b in names:
            aligned = pd.concat([returns[a], returns[b]], axis=1, join="inner").dropna()
            overlap.loc[a, b] = len(aligned)
            corr.loc[a, b] = (
                float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
                if len(aligned) > 2 else np.nan
            )

    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    body = []
    thin = False
    for a in names:
        cells = []
        for b in names:
            value = corr.loc[a, b]
            shared = int(overlap.loc[a, b] or 0)
            if a == b:
                cells.append("<td class='muted'>1.000</td>")
                continue
            if shared < MIN_CORRELATION_SESSIONS:
                thin = True
                cells.append(
                    f"<td class='warn' title='{shared} shared sessions'>"
                    f"{_fmt(value)}<sup>*</sup></td>"
                )
            else:
                cells.append(f"<td>{_fmt(value)}</td>")
        body.append(
            f"<tr><th class='rowkey'>{html.escape(a)}</th>{''.join(cells)}</tr>"
        )

    table = (
        f"<table><thead><tr><th></th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )
    if thin:
        table += (
            f"<p class='note'>* fewer than {MIN_CORRELATION_SESSIONS} overlapping "
            "sessions: the correlation is reported but should not be relied on.</p>"
        )
    return table


def regime_section(
    returns: pd.Series, benchmark_returns: pd.Series
) -> str:
    """Where the return came from, split by the benchmark's own regime."""
    if benchmark_returns is None or len(benchmark_returns) < 130:
        return "<p class='muted'>no benchmark returns to define regimes</p>"

    regimes = metrics.regime_returns(returns, benchmark_returns)
    if not regimes:
        return "<p class='muted'>not enough history to split by regime</p>"

    frame = pd.DataFrame(regimes).T.reset_index().rename(columns={"index": "regime"})
    for column in ("cagr", "vol", "max_dd"):
        if column in frame.columns:
            frame[column] = (frame[column] * 100).round(1)
    frame = frame.rename(columns={
        "cagr": "CAGR %", "vol": "Vol %", "max_dd": "Max DD %",
        "sharpe": "Sharpe", "days": "Sessions",
    })
    return frame_table(frame)


def versions_section(runs: Sequence[RunData]) -> str:
    """Current against previous, and whether the assumptions moved too.

    The trap this exists to catch: comparing two runs of the same strategy whose
    configs differ in cost, dates or capital is not a strategy comparison, and the
    difference will be attributed to the code change regardless. So the config
    diff is shown next to the metric diff.
    """
    if len(runs) < 2:
        return "<p class='muted'>only one run of this strategy on disk</p>"

    previous, current = runs[-2], runs[-1]
    rows = {
        f"previous · {previous.name}": stream_metrics(previous.equity),
        f"current · {current.name}": stream_metrics(current.equity),
    }
    body = metric_table(rows)

    watched = (
        "start_date", "end_date", "initial_capital", "benchmark",
        "universe_source", "price_adjustment", "params", "trial_index",
    )
    differences = []
    for key in watched:
        before, after = previous.config.get(key), current.config.get(key)
        if before != after:
            differences.append((key, before, after))

    for block in ("cost", "risk", "execution"):
        before = previous.config.get(block) or {}
        after = current.config.get(block) or {}
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                differences.append((f"{block}.{key}", before.get(key), after.get(key)))

    if differences:
        diff_rows = "".join(
            f"<tr><th class='rowkey'>{html.escape(k)}</th>"
            f"<td>{_fmt(b)}</td><td>{_fmt(a)}</td></tr>"
            for k, b, a in differences
        )
        body += (
            "<h3>Assumptions that also changed</h3>"
            "<table><thead><tr><th></th><th>previous</th><th>current</th></tr>"
            f"</thead><tbody>{diff_rows}</tbody></table>"
            "<p class='note'>The metric difference above cannot be attributed to "
            "the strategy alone while these differ.</p>"
        )
    else:
        body += (
            "<p class='note'>Configs match on dates, capital, benchmark, universe, "
            "costs, risk and execution, so the difference is the strategy.</p>"
        )

    return body


def research_sections(research_dir: Path) -> List[Tuple[str, str, str]]:
    """Cost scenarios, sweep and the verdict, pulled from a research run."""
    out: List[Tuple[str, str, str]] = []
    research_dir = Path(research_dir)

    payload_path = research_dir / "research.json"
    if payload_path.exists():
        payload = json.loads(payload_path.read_text())
        sections = payload.get("sections", {})

        verdict_rows = "".join(
            f"<tr><th class='rowkey'>{html.escape(name)}</th>"
            f"<td class='{ {'PASS':'ok','FAIL':'bad','MODERATE':'mid'}.get(body.get('status',''), 'muted') }'>"
            f"{html.escape(str(body.get('status')))}</td>"
            f"<td>{html.escape(str(body.get('headline','')))}</td></tr>"
            for name, body in sections.items()
        )
        out.append((
            f"Verdict: {payload.get('verdict', '—')}",
            f"<table><tbody>{verdict_rows}</tbody></table>",
            "From the research pass. NOT RUN never counts as a pass.",
        ))

        cost = (sections.get("Cost stress") or {}).get("detail", {})
        scenarios = cost.get("scenarios") or {}
        if scenarios:
            frame = pd.DataFrame(scenarios).T.reset_index().rename(
                columns={"index": "scenario"}
            )
            out.append((
                "Cost scenarios",
                frame_table(frame),
                "The same strategy at progressively punitive frictions.",
            ))

    sweep_path = research_dir / "sweep.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        highlight = None
        if "sharpe" in sweep.columns and len(sweep):
            highlight = int(np.nanargmax(sweep["sharpe"].to_numpy(dtype=float)))
        out.append((
            "Parameter sweep",
            frame_table(sweep, highlight=highlight),
            "Best Sharpe highlighted. A plateau matters more than the peak — an "
            "isolated bright cell is a fitting artifact.",
        ))

    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

CSS = """
:root { --ink:#1a1a1a; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; }
* { box-sizing:border-box; }
body { font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       color:var(--ink); background:var(--bg); margin:0; padding:32px 40px 72px; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:16px; margin:36px 0 8px; padding-bottom:6px;
     border-bottom:1px solid var(--line); }
h3 { font-size:13px; margin:20px 0 6px; color:var(--muted);
     text-transform:uppercase; letter-spacing:.04em; }
section { margin-bottom:8px; }
table { border-collapse:collapse; margin:8px 0 4px; font-variant-numeric:tabular-nums; }
th,td { padding:5px 12px; text-align:right; border-bottom:1px solid var(--line);
        white-space:nowrap; }
thead th { text-align:right; font-weight:600; color:var(--muted);
           font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
th.rowkey { text-align:left; font-weight:500; color:var(--ink); }
tr.mark td { background:#fff7cc; font-weight:600; }
.muted { color:var(--muted); }
.note { color:var(--muted); font-size:13px; margin:2px 0 10px; max-width:76ch; }
.ok { color:#15803d; font-weight:600; }
.bad { color:#b91c1c; font-weight:600; }
.mid { color:#b45309; font-weight:600; }
.warn { color:#b45309; }
.chart { width:100%; height:260px; border:1px solid var(--line);
         background:#fcfcfd; display:block; margin-top:8px; }
.legend { margin:6px 0 2px; font-size:13px; }
.key { margin-right:16px; white-space:nowrap; }
.key i { display:inline-block; width:10px; height:10px; margin-right:5px;
         border-radius:2px; vertical-align:middle; }
footer { margin-top:48px; padding-top:12px; border-top:1px solid var(--line);
         color:var(--muted); font-size:12px; }
"""


@dataclass
class ReportInputs:
    """Everything the report can draw on. All of it optional but the streams."""
    title: str
    runs: List[RunData] = field(default_factory=list)
    benchmark: Optional[pd.Series] = None
    benchmark_name: str = "SPY"
    equal_weight: Optional[pd.Series] = None
    is_oos_split: Optional[pd.Timestamp] = None
    research_dir: Optional[Path] = None
    version_history: List[RunData] = field(default_factory=list)


def build_html(inputs: ReportInputs) -> str:
    """Render the whole page."""
    runs = [r for r in inputs.runs if len(r.equity) > 2]
    if not runs:
        return "<html><body><p>No runs with an equity curve.</p></body></html>"

    common_start = max(r.equity.index[0] for r in runs)
    bench_returns = (
        inputs.benchmark.pct_change().dropna()
        if inputs.benchmark is not None and len(inputs.benchmark) > 2
        else None
    )

    # --- streams, all rebased to the common start ------------------------
    streams: Dict[str, pd.Series] = {
        r.name: _normalize(r.equity, common_start) for r in runs
    }
    if inputs.benchmark is not None:
        streams[inputs.benchmark_name] = _normalize(inputs.benchmark, common_start)
    if inputs.equal_weight is not None:
        streams["equal weight"] = _normalize(inputs.equal_weight, common_start)

    parts: List[str] = []

    # 1. Headline
    rows = {name: stream_metrics(series, bench_returns) for name, series in streams.items()}
    parts.append(section(
        "Headline",
        metric_table(rows),
        "Every stream rebased to 1.0 at the latest common start date "
        f"({common_start:%Y-%m-%d}), so books of different size compare directly. "
        "Metrics are recomputed here rather than read from each run's metrics.json, "
        "which may have been written by an older definition.",
    ))

    # 2. Equity and drawdown
    parts.append(section(
        "Equity curves",
        line_chart(streams, log_scale=True),
        "Log scale: equal vertical distance is equal percentage change, so a 20% "
        "move early reads the same as a 20% move late.",
    ))
    parts.append(section(
        "Drawdowns",
        line_chart(
            {n: metrics.drawdown_series(s) for n, s in streams.items()},
            zero_line=True,
        ),
        "The depth and, more importantly, the length. Time underwater is what "
        "capital actually reacts to.",
    ))

    # 3/4. Baselines, stated as differences rather than repeated levels
    baseline_names = [
        n for n in (inputs.benchmark_name, "equal weight") if n in streams
    ]
    if baseline_names:
        relative: Dict[str, Dict[str, Any]] = {}
        for run in runs:
            row: Dict[str, Any] = {}
            own = rows.get(run.name, {})
            for baseline in baseline_names:
                base = rows.get(baseline, {})
                for metric in ("CAGR", "Sharpe", "Max DD"):
                    mine, theirs = own.get(metric), base.get(metric)
                    if mine is None or theirs is None:
                        continue
                    row[f"{metric} − {baseline}"] = mine - theirs
            relative[run.name] = row

        parts.append(section(
            "Against the baselines",
            metric_table(relative),
            "Differences, not levels -- the levels are in the headline. Positive "
            f"means better than the baseline, except on Max DD where positive "
            f"means a shallower hole. {inputs.benchmark_name} is what you could "
            "have done with no work. Equal weight over the same traded names "
            "isolates selection skill from the universe's own drift: if it matches "
            "the strategy, the ranking is not adding anything.",
        ))
    else:
        parts.append(section(
            "Against the baselines",
            "<p class='muted'>no baseline available for this window</p>",
            "",
        ))

    # 5. In-sample vs out-of-sample
    if inputs.is_oos_split is not None:
        split = pd.Timestamp(inputs.is_oos_split)
        halves: Dict[str, Dict[str, Any]] = {}
        for run in runs:
            before = run.equity.loc[run.equity.index < split]
            after = run.equity.loc[run.equity.index >= split]
            if len(before) > 2:
                halves[f"{run.name} · in-sample"] = stream_metrics(before, bench_returns)
            if len(after) > 2:
                halves[f"{run.name} · out-of-sample"] = stream_metrics(after, bench_returns)
        parts.append(section(
            "In-sample versus out-of-sample",
            metric_table(halves),
            f"Split at {split:%Y-%m-%d}. A large gap is the signature of a "
            "strategy fitted to the first half.",
        ))

    # 6/7. From the research pass
    if inputs.research_dir is not None:
        for title, body, note in research_sections(inputs.research_dir):
            parts.append(section(title, body, note))

    # 8. Regimes and calendar years
    if bench_returns is not None:
        parts.append(section(
            "By market regime",
            regime_section(runs[0].returns, bench_returns),
            f"Regimes are defined by {inputs.benchmark_name}'s own trailing return "
            "and volatility, for the first stream only.",
        ))

    annual = {}
    for run in runs:
        yearly = metrics.annual_returns(run.equity)
        if len(yearly):
            annual[run.name] = {
                str(date.year): float(value) for date, value in yearly.items()
            }
    if annual:
        years = sorted({y for row in annual.values() for y in row})
        frame = pd.DataFrame(
            [{"year": y, **{n: annual[n].get(y) for n in annual}} for y in years]
        )
        for column in frame.columns[1:]:
            frame[column] = (frame[column] * 100).round(1)
        parts.append(section(
            "Calendar years (%)",
            frame_table(frame),
            "A single good year carrying a whole backtest is visible here and "
            "nowhere else.",
        ))

    # 9. Versions
    if inputs.version_history:
        parts.append(section(
            "Current versus previous version",
            versions_section(inputs.version_history),
            "",
        ))

    # 10. Correlation
    parts.append(section(
        "Correlation of daily returns",
        correlation_section(streams),
        "Computed on the overlapping window only. A strategy that correlates above "
        "about 0.8 with one you already run is mostly the same bet.",
    ))

    windows = "".join(
        f"<tr><th class='rowkey'>{html.escape(r.name)}</th>"
        f"<td>{html.escape(r.window)}</td>"
        f"<td>{html.escape(r.strategy_id)}</td></tr>"
        for r in runs
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(inputs.title)}</title><style>{CSS}</style></head>
<body>
<h1>{html.escape(inputs.title)}</h1>
<p class="muted">Generated {datetime.now():%Y-%m-%d %H:%M}</p>
<section><h2>Runs in this report</h2>
<table><thead><tr><th></th><th>window</th><th>strategy id</th></tr></thead>
<tbody>{windows}</tbody></table></section>
{"".join(parts)}
<footer>Built by src/backtest/report.py. Charts are inline SVG — no JavaScript,
no plotting library, one self-contained file.</footer>
</body></html>"""


def write_report(inputs: ReportInputs, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(inputs))
    logger.info("Wrote report to %s", out_path)
    return out_path
