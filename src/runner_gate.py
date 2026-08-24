"""runner_gate.py

DST-safe time gate for GitHub Actions cron jobs.

Because GitHub cron is UTC-only we schedule two triggers per workflow
(one for EST, one for EDT). This module checks whether the current
America/New_York time is inside the intended window and exits early
if not, preventing double-runs on DST transition days.

Usage (from workflow YAML):
    python src/runner_gate.py open   # exits 0 if ~9:30 AM ET, exits 0 with skip msg otherwise
    python src/runner_gate.py close  # exits 0 if ~4:30 PM ET

Exit codes:
    0 - always (so the workflow step succeeds). Prints SKIP or PROCEED.
    The calling workflow should check the step output to decide whether
    to continue.
"""

from __future__ import annotations

import sys
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

WINDOWS = {
    "open":  (9, 25, 9, 40),   # 9:25 - 9:40 AM ET
    "close": (16, 25, 16, 40), # 4:25 - 4:40 PM ET
}

TOLERANCE_MINUTES = 10


def check_window(window_name: str) -> bool:
    et = ZoneInfo("America/New_York")
    now = datetime.now(tz=et)

    if window_name not in WINDOWS:
        print(f"Unknown window '{window_name}'. Valid: {list(WINDOWS.keys())}")
        return False

    h_start, m_start, h_end, m_end = WINDOWS[window_name]
    start_min = h_start * 60 + m_start
    end_min = h_end * 60 + m_end
    now_min = now.hour * 60 + now.minute

    in_window = start_min <= now_min <= end_min
    print(
        f"runner_gate: window={window_name} now_et={now.strftime('%H:%M %Z')} "
        f"range={h_start}:{m_start:02d}-{h_end}:{m_end:02d} "
        f"-> {'PROCEED' if in_window else 'SKIP'}"
    )
    return in_window


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python runner_gate.py <open|close>")
        sys.exit(1)

    window = sys.argv[1].lower()
    in_window = check_window(window)

    if in_window:
        print("GATE_RESULT=proceed")
    else:
        print("GATE_RESULT=skip")


if __name__ == "__main__":
    main()
