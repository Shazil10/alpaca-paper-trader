"""Partition provenance: which provider wrote each year of the lake.

    data/prices/manifest.json

The lake schema stays stable -- no ``source`` column on every row. Provenance is
recorded once per year partition instead, which is enough because of the rule it
enforces: **a year holds one provider**. ``writer.persist`` checks the manifest
before every write and refuses one that would put a second provider's rows into a
year owned by another. So "which provider produced this bar" always has a single
answer: look up its year.

Per partition::

    year, file, provider, feed, adjustment, downloaded_at,
    start_date, end_date, row_count, symbol_count, content_hash, notes

``content_hash`` is a SHA-256 over the partition's rows serialized exactly as the
hot-year CSV writer would, so it is independent of file format and of Parquet
library versions. Two rebuilds from the same provider data hash identically;
anything that alters a single price changes it.

Writes without provenance
-------------------------
A write that names no provider is recorded as ``"unspecified"``. That keeps
low-level code and tests working, and it cannot slip into the real lake: an
``unspecified`` write into a year owned by ``alpaca`` or ``yahoo`` is a provider
mismatch like any other and is refused. Nothing enters silently; the worst case
is a year whose manifest entry says, in plain text, that nobody knows.

Exceptions
----------
A secondary source can fill a genuine gap only through ``quarantine`` and a
reconciliation record, and when it does the partition gains an explicit
``exceptions`` entry naming the symbol, the date range and the provider. Mixed is
allowed; *quietly* mixed is not.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data_pipeline import schema

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
UNSPECIFIED = "unspecified"


class MixedProviderError(ValueError):
    """A write would put a second provider's rows into a single-provider year."""


@dataclass(frozen=True)
class Provenance:
    provider: str = UNSPECIFIED
    feed: str = ""
    adjustment: str = ""

    @classmethod
    def coerce(cls, value: Any) -> "Provenance":
        if value is None:
            return cls()
        if isinstance(value, Provenance):
            return value
        if isinstance(value, dict):
            return cls(
                provider=str(value.get("provider") or UNSPECIFIED),
                feed=str(value.get("feed") or ""),
                adjustment=str(value.get("adjustment") or ""),
            )
        raise TypeError(f"cannot interpret provenance {value!r}")


# ---------------------------------------------------------------------------
# Location, load, save
# ---------------------------------------------------------------------------

def manifest_path(root: Optional[Path] = None) -> Path:
    """``data/prices/manifest.json`` for the canonical lake; see ``schema.sidecar_dir``."""
    return schema.sidecar_dir(root) / MANIFEST_NAME


def load(root: Optional[Path] = None) -> Dict[str, Any]:
    path = manifest_path(root)
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "partitions": {}}
    data = json.loads(path.read_text())
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("partitions", {})
    return data


def save(data: Dict[str, Any], root: Optional[Path] = None) -> Path:
    """Write with sorted keys and a trailing newline, so git diffs stay small."""
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {
        "schema_version": data.get("schema_version", SCHEMA_VERSION),
        "partitions": {
            str(k): data["partitions"][k]
            for k in sorted(data.get("partitions", {}), key=lambda y: int(y))
        },
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ordered, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def partition(year: int, root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    return load(root)["partitions"].get(str(int(year)))


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def content_hash(frame: pd.DataFrame) -> str:
    """SHA-256 over the rows in canonical CSV form. Format-independent."""
    rows = schema.coerce(frame)
    buffer = io.StringIO()
    rows.to_csv(
        buffer,
        index=False,
        columns=schema.COLUMNS,
        float_format=schema.CSV_FLOAT_FORMAT,
        date_format=schema.CSV_DATE_FORMAT,
        lineterminator=schema.CSV_LINE_TERMINATOR,
    )
    return "sha256:" + hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The single-provider rule
# ---------------------------------------------------------------------------

def check_write(
    year: int,
    provenance: Any,
    root: Optional[Path] = None,
    *,
    file_exists: bool = False,
) -> None:
    """Raise if writing ``provenance`` rows into ``year`` would mix providers.

    A year that exists on disk but has no manifest entry is of unknown origin.
    Writing a *named* provider into it would stamp every existing row with that
    provider -- the quiet mixing this module exists to stop -- so that is refused,
    and the message says how to bootstrap the entry. Writing ``unspecified`` into
    it is allowed, because the resulting entry says exactly what is known: nothing.
    """
    incoming = Provenance.coerce(provenance)
    entry = partition(year, root)

    if entry is None:
        if file_exists and incoming.provider != UNSPECIFIED:
            raise MixedProviderError(
                f"{year}: partition exists on disk with no manifest entry, so its "
                f"provider is unknown. Record it first with "
                f"manifest.bootstrap(...) before writing {incoming.provider!r} rows."
            )
        return

    owner = str(entry.get("provider") or UNSPECIFIED)
    if owner != incoming.provider:
        raise MixedProviderError(
            f"{year}: partition is owned by {owner!r}; refusing to write "
            f"{incoming.provider!r} rows into it. A year holds one provider. "
            f"Replace the whole partition, or route the secondary data through "
            f"quarantine and a reconciliation exception."
        )


def record(
    year: int,
    frame: pd.DataFrame,
    provenance: Any,
    root: Optional[Path] = None,
    *,
    file_name: Optional[str] = None,
    downloaded_at: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Write (or refresh) the manifest entry for ``year`` from its full contents."""
    source = Provenance.coerce(provenance)
    rows = schema.coerce(frame)
    data = load(root)
    key = str(int(year))
    previous = data["partitions"].get(key, {})

    entry = {
        "year": int(year),
        "file": file_name or previous.get("file"),
        "provider": source.provider,
        "feed": source.feed,
        "adjustment": source.adjustment,
        "downloaded_at": downloaded_at or _utc_now(),
        "start_date": _date_or_none(rows[schema.DATE].min()) if len(rows) else None,
        "end_date": _date_or_none(rows[schema.DATE].max()) if len(rows) else None,
        "row_count": int(len(rows)),
        "symbol_count": int(rows[schema.SYMBOL].nunique()) if len(rows) else 0,
        "content_hash": content_hash(rows),
        "notes": list(notes if notes is not None else previous.get("notes", [])),
        "exceptions": list(previous.get("exceptions", [])),
    }
    data["partitions"][key] = entry
    save(data, root)
    return entry


def bootstrap(
    year: int,
    provenance: Any,
    root: Optional[Path] = None,
    *,
    downloaded_at: Optional[str] = None,
    notes: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Record provenance for a partition written before the manifest existed.

    Refuses to overwrite an existing entry unless asked, so a mistaken bootstrap
    cannot relabel a partition that already has a known owner.
    """
    existing = partition(year, root)
    if existing is not None and not overwrite:
        raise MixedProviderError(
            f"{year}: manifest already records provider "
            f"{existing.get('provider')!r}; pass overwrite=True to replace it"
        )
    path = schema.resolve_year_path(int(year), root)
    if path is None:
        raise FileNotFoundError(f"no partition on disk for {year}")
    frame = schema.read_frame(path)
    return record(
        year, frame, provenance, root,
        file_name=path.name, downloaded_at=downloaded_at, notes=notes,
    )


def add_exception(
    year: int,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    provider: str,
    reason: str,
    reconciliation: str,
    root: Optional[Path] = None,
) -> None:
    """Record that a (symbol, date range) inside ``year`` came from elsewhere."""
    data = load(root)
    entry = data["partitions"].get(str(int(year)))
    if entry is None:
        raise KeyError(f"no manifest entry for {year}")
    entry.setdefault("exceptions", []).append({
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "provider": provider,
        "reason": reason,
        "reconciliation": reconciliation,
        "recorded_at": _utc_now(),
    })
    save(data, root)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(root: Optional[Path] = None) -> List[str]:
    """Problems between the manifest and the files on disk. Empty means clean."""
    problems: List[str] = []
    data = load(root)
    recorded = {int(k): v for k, v in data["partitions"].items()}

    for path in schema.discover_year_files(root):
        year = int(path.stem)
        entry = recorded.get(year)
        if entry is None:
            problems.append(f"{year}: on disk but absent from the manifest")
            continue
        frame = schema.read_frame(path)
        actual = content_hash(frame)
        if actual != entry.get("content_hash"):
            problems.append(
                f"{year}: content hash mismatch (manifest {entry.get('content_hash')}, "
                f"disk {actual}) — the partition changed without a manifest update"
            )
        if int(entry.get("row_count", -1)) != len(frame):
            problems.append(
                f"{year}: row count {len(frame)} on disk vs {entry.get('row_count')} "
                f"in manifest"
            )
        if entry.get("provider") in (None, "", UNSPECIFIED):
            problems.append(f"{year}: provider is unspecified")

    on_disk = {int(p.stem) for p in schema.discover_year_files(root)}
    for year in sorted(set(recorded) - on_disk):
        problems.append(f"{year}: in the manifest but no partition on disk")

    return problems


def providers_by_year(root: Optional[Path] = None) -> Dict[int, str]:
    return {
        int(k): str(v.get("provider") or UNSPECIFIED)
        for k, v in load(root)["partitions"].items()
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _date_or_none(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")
