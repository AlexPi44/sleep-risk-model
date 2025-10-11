#!/usr/bin/env python3
"""check_nhanes_downloads.py 

Download NHANES XPT files for cycles 2005-2016, convert to CSV, and report variable presence.
Idempotent: safe to re-run.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List
import requests
from tqdm import tqdm

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except Exception:
    HAS_PYREADSTAT = False
    print("WARNING: pyreadstat not installed. Install with: pip install pyreadstat", file=sys.stderr)

import pandas as pd

CYCLES = [
    "2005-2006",
    "2007-2008",
    "2009-2010",
    "2011-2012",
    "2013-2014",
    "2015-2016",
]

CYCLE_SUFFIX = {
    "2005-2006": "D",
    "2007-2008": "E",
    "2009-2010": "F",
    "2011-2012": "G",
    "2013-2014": "H",
    "2015-2016": "I",
}

WANTED_FILES = [
    "DEMO",
    "SLQ",
    "PAQ",
    "DR1TOT",
    "DR2TOT",
    "BMX",
    "BPX",
    "DPQ",
    "ALQ",
    "SMQ",
]

KEY_VARIABLES = {
    "SLQ": ["SLQ060", "SLQ050", "SLQ120", "SLD012"],
    "PAQ": ["PAD680", "PAD700"],
    "DR1TOT": ["DR1TKCAL", "DR1TFIBE", "DR1TSUGR", "DR1TCAFF"],
    "DR2TOT": ["DR2TKCAL", "DR2TFIBE", "DR2TSUGR", "DR2TCAFF"],
    "BMX": ["BMXBMI"],
    "BPX": ["BPXSY1", "BPXDI1"],
    "DPQ": [f"DPQ0{i}0" for i in range(1, 10)],
    "ALQ": ["ALQ130"],
    "SMQ": ["SMQ040"],
}

BASE_URL = "https://wwwn.cdc.gov/nchs/nhanes"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_SLEEP = 2.0


def download_file(url: str, dest: Path) -> bool:
    """Download file with retries and progress bar."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code == 200:
                    total = int(r.headers.get("content-length", 0) or 0)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {dest.name}",
                        disable=False,
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                    return True
                elif r.status_code == 404:
                    print(f"NOT FOUND: {url}", file=sys.stderr)
                    return False
                else:
                    print(f"HTTP {r.status_code}: {url}", file=sys.stderr)
                    time.sleep(RETRY_SLEEP * attempt)
        except Exception as e:
            print(
                f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}",
                file=sys.stderr,
            )
            time.sleep(RETRY_SLEEP * attempt)
    return False


def read_xpt_variables(xpt_path: Path) -> List[str]:
    """Read variable names from XPT file. Uses pyreadstat (required)."""
    xpt_path_str = str(xpt_path)
    if not xpt_path.exists():
        raise FileNotFoundError(f"XPT file not found: {xpt_path_str}")

    if not HAS_PYREADSTAT:
        raise RuntimeError(
            "pyreadstat is required to read XPT files. Install with: pip install pyreadstat"
        )

    try:
        df, meta = pyreadstat.read_xport(xpt_path_str)
        return list(df.columns)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read XPT {xpt_path}: {e}. Ensure file is valid XPT format."
        )


def convert_xpt_to_csv(xpt_path: Path, csv_path: Path) -> None:
    """Convert XPT to CSV using pyreadstat."""
    if not HAS_PYREADSTAT:
        raise RuntimeError("pyreadstat required for conversion")

    try:
        df, meta = pyreadstat.read_xport(str(xpt_path))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(csv_path), index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to convert {xpt_path} to CSV: {e}")


def process_cycle(cycle: str, out_dir: Path, convert: bool = True) -> Dict:
    """Download and process a single NHANES cycle."""
    suffix = CYCLE_SUFFIX.get(cycle)
    if not suffix:
        raise ValueError(f"No suffix mapping for cycle {cycle}")

    cycle_base = f"{BASE_URL}/{cycle}"
    cycle_dir = out_dir / cycle.replace("-", "_")
    report = {"cycle": cycle, "files": {}}

    print(f"\nProcessing cycle {cycle}...")

    for basefile in WANTED_FILES:
        filename = f"{basefile}_{suffix}.XPT"
        url = f"{cycle_base}/{filename}"
        dest = cycle_dir / filename

        file_info = {
            "url": url,
            "downloaded": False,
            "variables": None,
            "missing_variables": None,
            "csv": None,
        }

        # Check if already downloaded
        if dest.exists() and dest.stat().st_size > 0:
            file_info["downloaded"] = True
            print(f"  Found existing: {dest.name}")
        else:
            ok = download_file(url, dest)
            if not ok:
                print(f"  SKIP: {filename} (not found or failed)")
                report["files"][basefile] = file_info
                continue
            file_info["downloaded"] = True

        # Read and check variables
        try:
            cols = read_xpt_variables(dest)
            file_info["variables"] = cols
            file_info["present_count"] = len(cols)
        except Exception as e:
            print(f"  ERROR reading {filename}: {e}", file=sys.stderr)
            file_info["error"] = str(e)
            report["files"][basefile] = file_info
            continue

        # Check for missing key variables
        want_vars = KEY_VARIABLES.get(basefile, [])
        missing = [v for v in want_vars if v not in cols]
        file_info["missing_variables"] = missing

        if missing:
            print(f"  {filename}: missing {missing}")

        # Convert to CSV if requested
        if convert:
            csv_name = filename.replace(".XPT", ".csv")
            csv_path = cycle_dir / "csv" / csv_name
            try:
                if not csv_path.exists():
                    convert_xpt_to_csv(dest, csv_path)
                    print(f"  Converted: {csv_path.name}")
                else:
                    print(f"  CSV exists: {csv_path.name}")
                file_info["csv"] = str(csv_path)
            except Exception as e:
                print(f"  ERROR converting {filename}: {e}", file=sys.stderr)
                file_info["csv_error"] = str(e)

        report["files"][basefile] = file_info

    return report


def main(args):
    """Main entry point."""
    out_dir = Path(args.out)
    report_dir = Path(args.report)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_PYREADSTAT:
        print(
            "ERROR: pyreadstat is required. Install with: pip install pyreadstat",
            file=sys.stderr,
        )
        sys.exit(1)

    overall = {
        "runs": [],
        "pyreadstat_available": HAS_PYREADSTAT,
        "timestamp": str(Path.cwd()),
    }

    for cycle in CYCLES:
        try:
            rep = process_cycle(cycle, out_dir, convert=args.convert)
            overall["runs"].append(rep)
            report_path = report_dir / f"{cycle.replace('-', '_')}_report.json"
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=2)
            print(f"Saved report: {report_path}")
        except Exception as e:
            print(f"ERROR processing {cycle}: {e}", file=sys.stderr)
            overall["runs"].append({"cycle": cycle, "error": str(e)})

    agg_path = report_dir / "aggregate_report.json"
    with open(agg_path, "w", encoding="utf-8") as fh:
        json.dump(overall, fh, indent=2)
    print(f"\nAggregate report saved: {agg_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download NHANES cycles & check variable presence"
    )
    parser.add_argument(
        "--out", default="data/raw", help="Root output directory for downloaded XPTs"
    )
    parser.add_argument(
        "--report", default="reports", help="Directory for JSON reports"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        default=True,
        help="Also convert XPT -> CSV",
    )
    args = parser.parse_args()
    main(args)
