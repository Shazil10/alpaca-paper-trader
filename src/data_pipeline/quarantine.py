"""Secondary-source bars, held apart from the canonical lake until reconciled.

    data/prices/quarantine/<provider>/<label>.csv                  the bars
    data/prices/quarantine/<provider>/<label>.json                 who, when, why
    data/prices/quarantine/<provider>/<label>.reconciliation.json  the evidence

The canonical lake has one provider per year (``manifest.py``). There are still
real gaps a second source can close -- a delisted name Alpaca never carried, a
session it is missing -- and the question is how to let those rows in without
the lake quietly becoming "mostly Alpaca, partly whoever".

The answer is a holding area and a gate:

``hold``
    Writes a secondary source's bars here, in the lake schema, with a metadata
    file naming the provider, the reason, and a content hash. Nothing reads
    this directory by default. The Phase 7 pilot's Yahoo and Stooq comparison
    frames live here too, for the same reason: they must never be mistaken for
    canonical data.

``reconcile``
    Compares held bars with the canonical bars *around* the gap, per symbol:
    enough overlapping sessions, daily returns that agree, and a price-scale
    ratio that is the same on both sides. A split or a different adjustment
    convention inside the gap shows up as an inconsistent ratio and fails.

``promote``
    Only for symbols that reconcile. Fills (date, symbol) keys the canonical
    lake does *not* have -- it never overwrites a canonical bar -- after
    rescaling the held prices onto the canonical scale by the measured ratio.
    Each promotion adds an explicit ``exceptions`` entry to the year's manifest
    record: symbol, date range, provider, reason, and the path of the
    reconciliation evidence. Mixed is allowed; quietly mixed is not.

Nothing calls ``promote`` automatically. The decision rule in
``scripts/classify_gaps.py`` is: rebuild on Alpaca, classify what is still
missing, and only then trial a secondary source on the specific cases where
the classification says one could help.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_pipeline import anchor, manifest, schema, store

logger = logging.getLogger(__name__)

QUARANTINE_SUBDIR = "quarantine"

#: Overlapping sessions with the canonical series a symbol needs before its
#: held bars can be trusted to sit on the same scale.
MIN_OVERLAP = 10

#: Share of overlapping daily returns allowed to differ by more than
#: ``RETURN_TOLERANCE`` between sources.
MAX_DISAGREEMENT_RATE = 0.05
RETURN_TOLERANCE = 0.005

#: The canonical/held price ratio must be this stable across the overlap
#: (max/min - 1). A split or a different adjustment inside it breaks this.
MAX_RATIO_DISPERSION = 0.01

#: Canonical context fetched around the held rows for the comparison.
CONTEXT_DAYS = 45


class ReconciliationError(ValueError):
    """No held symbol reconciled against the canonical lake."""


def quarantine_dir(root: Optional[Path] = None) -> Path:
    return schema.sidecar_dir(root) / QUARANTINE_SUBDIR


def _paths(provider: str, label: str, root: Optional[Path]) -> Tuple[Path, Path, Path]:
    base = quarantine_dir(root) / provider
    return (base / f"{label}.csv", base / f"{label}.json",
            base / f"{label}.reconciliation.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Hold
# ---------------------------------------------------------------------------

def hold(
    frame: pd.DataFrame,
    provider: str,
    label: str,
    *,
    root: Optional[Path] = None,
    reason: str = "",
    provenance: Optional[dict] = None,
) -> Path:
    """Write secondary-source bars to quarantine. Never touches the lake."""
    bars = schema.coerce(frame)
    data_path, meta_path, _ = _paths(provider, label, root)
    schema.write_csv(bars, data_path)
    meta = {
        "provider": provider,
        "label": label,
        "reason": reason,
        "provenance": provenance or {"provider": provider},
        "held_at": _utc_now(),
        "rows": int(len(bars)),
        "symbols": sorted(bars[schema.SYMBOL].unique().tolist()) if len(bars) else [],
        "start_date": str(bars[schema.DATE].min().date()) if len(bars) else None,
        "end_date": str(bars[schema.DATE].max().date()) if len(bars) else None,
        "content_hash": manifest.content_hash(bars),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    logger.info("held %d %s row(s) in quarantine as %s", len(bars), provider, data_path)
    return data_path


def load_held(provider: str, label: str, root: Optional[Path] = None) -> Tuple[pd.DataFrame, dict]:
    data_path, meta_path, _ = _paths(provider, label, root)
    if not data_path.exists():
        raise FileNotFoundError(f"nothing held as {provider}/{label}")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    frame = schema.read_frame(data_path)
    if meta.get("content_hash") and manifest.content_hash(frame) != meta["content_hash"]:
        raise ValueError(f"{provider}/{label}: held bars changed since they were held")
    return frame, meta


def list_held(root: Optional[Path] = None) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(quarantine_dir(root).glob("*/*.json")):
        if meta_path.name.endswith(".reconciliation.json"):
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError:
            continue
        rows.append({k: meta.get(k) for k in
                     ("provider", "label", "reason", "held_at", "rows", "start_date", "end_date")})
    return pd.DataFrame(rows, columns=["provider", "label", "reason", "held_at",
                                       "rows", "start_date", "end_date"])


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------

def reconcile(held: pd.DataFrame, canonical: pd.DataFrame) -> Dict[str, dict]:
    """Per-symbol verdicts: can this symbol's held bars fill canonical gaps?"""
    key = [schema.DATE, schema.SYMBOL]
    out: Dict[str, dict] = {}
    held = schema.coerce(held)
    canonical = schema.coerce(canonical)
    canon_keys = set(map(tuple, canonical[key].astype(str).values.tolist()))

    for symbol, rows in held.groupby(schema.SYMBOL, sort=True):
        canon = canonical[canonical[schema.SYMBOL] == symbol]
        both = rows.merge(canon, on=key, suffixes=("_held", "_canon"))
        fills = rows[[tuple(k) not in canon_keys for k in rows[key].astype(str).values.tolist()]]
        verdict: Dict[str, object] = {
            "overlap": int(len(both)),
            "fill_rows": int(len(fills)),
            "fill_start": str(fills[schema.DATE].min().date()) if len(fills) else None,
            "fill_end": str(fills[schema.DATE].max().date()) if len(fills) else None,
            "reasons": [],
        }
        reasons: List[str] = verdict["reasons"]  # type: ignore[assignment]

        if len(fills) == 0:
            reasons.append("nothing to fill: every held key already exists canonically")
        if len(both) < MIN_OVERLAP:
            reasons.append(f"only {len(both)} overlapping session(s), need {MIN_OVERLAP}")
        else:
            adj_ratio = both["adj_close_canon"] / both["adj_close_held"]
            close_ratio = both["close_canon"] / both["close_held"]
            for name, ratio in (("adj_close", adj_ratio), ("close", close_ratio)):
                dispersion = float(ratio.max() / ratio.min() - 1.0)
                verdict[f"{name}_scale"] = float(np.median(ratio))
                verdict[f"{name}_dispersion"] = dispersion
                if dispersion > MAX_RATIO_DISPERSION:
                    reasons.append(
                        f"{name} ratio varies {dispersion:.2%} across the overlap "
                        f"(a split or a different adjustment convention)"
                    )
            # Returns only over the shared dates, so a hole in either series
            # cannot pair a one-day return with a multi-day one.
            shared = both.sort_values(schema.DATE)
            pairs = pd.DataFrame({
                "held": shared["adj_close_held"].pct_change().values,
                "canon": shared["adj_close_canon"].pct_change().values,
            }).dropna()
            rate = float(((pairs["held"] - pairs["canon"]).abs() > RETURN_TOLERANCE).mean()) \
                if len(pairs) else 1.0
            verdict["return_disagreement_rate"] = rate
            if rate > MAX_DISAGREEMENT_RATE:
                reasons.append(
                    f"{rate:.1%} of overlapping returns disagree by more than "
                    f"{RETURN_TOLERANCE:.1%}"
                )
        verdict["ok"] = not reasons
        out[str(symbol)] = verdict
    return out


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------

def promote(
    provider: str,
    label: str,
    *,
    reason: str,
    root: Optional[Path] = None,
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Fill canonical gaps from held bars for symbols that reconcile.

    Raises ``ReconciliationError`` if none do, leaving the lake untouched.
    """
    held, meta = load_held(provider, label, root)
    if symbols is not None:
        wanted = {str(s).upper() for s in symbols}
        held = held[held[schema.SYMBOL].isin(wanted)]
    if len(held) == 0:
        raise ReconciliationError(f"{provider}/{label}: no held rows to promote")

    start = held[schema.DATE].min() - pd.Timedelta(days=CONTEXT_DAYS)
    end = held[schema.DATE].max() + pd.Timedelta(days=CONTEXT_DAYS)
    canonical = store.load_prices(sorted(held[schema.SYMBOL].unique()), start, end, root=root)

    verdicts = reconcile(held, canonical)
    _, _, evidence_path = _paths(provider, label, root)
    evidence_path.write_text(json.dumps(
        {"provider": provider, "label": label, "reason": reason,
         "reconciled_at": _utc_now(), "symbols": verdicts},
        indent=2, sort_keys=True, default=str,
    ) + "\n")

    passing = sorted(s for s, v in verdicts.items() if v["ok"])
    if not passing:
        raise ReconciliationError(
            f"{provider}/{label}: no symbol reconciled; see {evidence_path}"
        )

    key = [schema.DATE, schema.SYMBOL]
    canon_keys = set(map(tuple, canonical[key].astype(str).values.tolist()))
    fills = []
    for symbol in passing:
        rows = held[held[schema.SYMBOL] == symbol]
        rows = rows[[tuple(k) not in canon_keys for k in rows[key].astype(str).values.tolist()]].copy()
        close_scale = float(verdicts[symbol]["close_scale"])
        adj_scale = float(verdicts[symbol]["adj_close_scale"])
        for col in schema.RAW_PRICE_COLUMNS:
            rows[col] = rows[col] * close_scale
        rows[schema.ADJ_CLOSE] = rows[schema.ADJ_CLOSE] * adj_scale
        if close_scale > 0:
            rows[schema.VOLUME] = (rows[schema.VOLUME].astype("float64") / close_scale).round()
        fills.append(rows)
    fill = schema.coerce(pd.concat(fills, ignore_index=True))

    pending = anchor.load(root)
    promoted_years: List[int] = []
    for year in schema.years_in(fill):
        path = schema.resolve_year_path(year, root)
        if path is None:
            raise manifest.MixedProviderError(
                f"{year}: no canonical partition to fill; a quarantined source "
                f"cannot create a year on its own"
            )
        entry = manifest.partition(year, root)
        if entry is None:
            raise manifest.MixedProviderError(f"{year}: partition has no manifest owner")
        existing = anchor.apply(schema.read_frame(path), year, pending)
        rows = fill[fill[schema.DATE].dt.year == year]
        merged = schema.upsert(existing, rows)
        schema.write_year(merged, year, root, hot=path.suffix == ".csv")
        manifest.record(year, merged, entry, root, file_name=path.name,
                        downloaded_at=entry.get("downloaded_at"))
        for symbol, group in rows.groupby(schema.SYMBOL):
            manifest.add_exception(
                year, symbol=str(symbol),
                start_date=str(group[schema.DATE].min().date()),
                end_date=str(group[schema.DATE].max().date()),
                provider=provider, reason=reason,
                reconciliation=str(evidence_path), root=root,
            )
        promoted_years.append(year)
    anchor.save(anchor.clear(pending, promoted_years), root)

    logger.info(
        "promoted %d row(s) for %d symbol(s) from %s/%s into year(s) %s",
        len(fill), len(passing), provider, label, promoted_years,
    )
    return {
        "promoted_symbols": passing,
        "refused_symbols": sorted(set(verdicts) - set(passing)),
        "rows": int(len(fill)),
        "years": promoted_years,
        "evidence": str(evidence_path),
    }
