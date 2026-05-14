"""CSV logging helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_result_row(csv_path: str | Path, row: dict) -> None:
    """Append one experiment result row to a CSV file."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)

