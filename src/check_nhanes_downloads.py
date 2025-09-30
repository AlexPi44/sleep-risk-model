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

try:
    import xport
    HAS_XPORT = True
except Exception:
    HAS_XPORT = False

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

BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_SLEEP = 2.0

def download_file(url: str, dest: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                if r.status_code == 200:
                    total = int(r.headers.get("content-length", 0) or 0)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                    return True
                else:
                    if r.status_code == 404:
                        return False
            time.sleep(RETRY_SLEEP * attempt)
        except Exception as e:
            print(f"Attempt {attempt} failed for {url}: {e}", file=sys.stderr)
            time.sleep(RETRY_SLEEP * attempt)
    return False

def read_xpt_variables(xpt_path: Path) -> List[str]:
    xpt_path_str = str(xpt_path)
    if not xpt_path.exists():
        raise FileNotFoundError(xpt_path_str)

    if HAS_PYREADSTAT:
        try:
            df, meta = pyreadstat.read_xport(xpt_path_str)
            return list(df.columns)
        except Exception as e:
            print(f\"pyreadstat read failed for {xpt_path}: {e}\", file=sys.stderr)

    try:
        df = pd.read_sas(xpt_path_str, format=\"xport\", encoding=\"utf-8\")
        return list(df.columns)
    except Exception as e:
        print(f\"pandas.read_sas failed for {xpt_path}: {e}\", file=sys.stderr)

    if HAS_XPORT:
        try:
            with open(xpt_path_str, \"rb\") as fh:
                library = xport.Unmarshaller(fh).load()
                cols = []
                for table in library.values():
                    if hasattr(table, \"to_pandas\"):
                        df0 = table.to_pandas()
                        cols = list(df0.columns)
                        break
                if cols:
                    return cols
        except Exception as e:
            print(f\"xport fallback failed for {xpt_path}: {e}\", file=sys.stderr)

    raise RuntimeError(f\"Could not read XPT: {xpt_path} (no supported reader succeeded)\")

def convert_xpt_to_csv(xpt_path: Path, csv_path: Path) -> None:
    if HAS_PYREADSTAT:
        df, meta = pyreadstat.read_xport(str(xpt_path))
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(csv_path), index=False)
    else:
        df = pd.read_sas(str(xpt_path), format=\"xport\", encoding=\"utf-8\")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(csv_path), index=False)

def process_cycle(cycle: str, out_dir: Path, convert: bool = True) -> Dict:
    suffix = CYCLE_SUFFIX.get(cycle)
    if not suffix:
        raise ValueError(f\"No suffix mapping for cycle {cycle}\")

    cycle_base = f\"{BASE_URL}/{cycle}\"
    cycle_dir = out_dir / cycle.replace(\"-\", \"_\")
    report = {\"cycle\": cycle, \"files\": {}}

    for basefile in WANTED_FILES:
        filename = f\"{basefile}_{suffix}.XPT\"
        url = f\"{cycle_base}/{filename}\"
        dest = cycle_dir / filename

        file_info = {\"url\": url, \"downloaded\": False, \"variables\": None, \"missing_variables\": None, \"csv\": None}
        if dest.exists() and dest.stat().st_size > 0:
            file_info[\"downloaded\"] = True
            print(f\"Found existing file: {dest}\")
        else:
            ok = download_file(url, dest)
            if not ok:
                file_info[\"downloaded\"] = False
                report[\"files\"][basefile] = file_info
                continue
            else:
                file_info[\"downloaded\"] = True

        try:
            cols = read_xpt_variables(dest)
            file_info[\"variables\"] = cols
        except Exception as e:
            file_info[\"variables\"] = None
            file_info[\"error\"] = str(e)
            report[\"files\"][basefile] = file_info
            continue

        want_vars = KEY_VARIABLES.get(basefile, [])
        missing = [v for v in want_vars if v not in cols]
        file_info[\"missing_variables\"] = missing
        file_info[\"present_count\"] = len(cols)

        if convert:
            csv_name = filename.replace(\".XPT\", \".csv\")
            csv_path = cycle_dir / \"csv\" / csv_name
            try:
                convert_xpt_to_csv(dest, csv_path)
                file_info[\"csv\"] = str(csv_path)
            except Exception as e:
                file_info[\"csv_error\"] = str(e)

        report[\"files\"][basefile] = file_info

    return report

def main(args):
    out_dir = Path(args.out)
    report_dir = Path(args.report)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    overall = {\"runs\": [], \"pyreadstat\": HAS_PYREADSTAT, \"xport_available\": HAS_XPORT}

    for cycle in CYCLES:
        print(f\"\\nProcessing cycle {cycle} ...\")
        try:
            rep = process_cycle(cycle, out_dir, convert=args.convert)
            overall[\"runs\"].append(rep)
            report_path = report_dir / f\"{cycle.replace('-', '_')}_report.json\"
            with open(report_path, \"w\", encoding=\"utf-8\") as fh:
                json.dump(rep, fh, indent=2)
            print(f\"Saved report: {report_path}\")
        except Exception as e:
            print(f\"ERROR processing {cycle}: {e}\", file=sys.stderr)

    agg_path = report_dir / \"aggregate_report.json\"
    with open(agg_path, \"w\", encoding=\"utf-8\") as fh:
        json.dump(overall, fh, indent=2)
    print(f\"\\nAggregate report saved: {agg_path}\")\n    print(\"Done.\")\n\nif __name__ == \"__main__\":\n    parser = argparse.ArgumentParser(description=\"Download NHANES cycles & check variable presence\")\n    parser.add_argument(\"--out\", default=\"data/raw\", help=\"Root output directory for downloaded XPTs\")\n    parser.add_argument(\"--report\", default=\"reports\", help=\"Directory for JSON reports\")\n    parser.add_argument(\"--convert\", action=\"store_true\", default=True, help=\"Also convert XPT -> CSV\")\n    args = parser.parse_args()\n    main(args)\n
