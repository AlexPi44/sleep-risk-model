#!/usr/bin/env python3
"""
check_nhanes_downloads.py

Robust downloader + validator for NHANES XPT files.

This version uses the CDC Public DataFiles path (Nchs/Data/Nhanes/Public/<year>/DataFiles/<file>.xpt)
and validates downloads with pyreadstat. It also saves suspicious HTML responses as .bad.html.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import requests
import pyreadstat

# --- Configuration / default datasets & mapping ---
DEFAULT_CYCLES = [
    "2005_2006", "2007_2008", "2009_2010",
    "2011_2012", "2013_2014", "2015_2016"
]
# mapping cycle -> NHANES letter suffix used in filenames
CYCLE_LETTER = {
    "2005_2006": "D",
    "2007_2008": "E",
    "2009_2010": "F",
    "2011_2012": "G",
    "2013_2014": "H",
    "2015_2016": "I",
}

# datasets we want for each cycle (base names; .XPT suffix added)
DATASETS = [
    "DEMO", "SLQ", "PAQ", "DR1TOT", "DR2TOT",
    "BMX", "BPX", "DPQ", "ALQ", "SMQ"
]

# IMPORTANT: use the CDC Public DataFiles path (year subfolder)
NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

# Minimum plausible XPT size (bytes) to flag tiny responses (<2 KB likely HTML/err page)
MIN_VALID_SIZE = 2048

# HTTP headers to appear like a normal browser
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/x-sas, application/octet-stream, */*"
}


def build_filename(base: str, letter: str) -> str:
    return f"{base}_{letter}.XPT"


def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
    tmp.replace(path)


def looks_like_html_file(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            start = fh.read(2048).lower()
            return b"<html" in start or b"<!doctype" in start or b"<head" in start
    except Exception:
        return False


def validate_xpt(path: Path) -> Tuple[bool, Optional[str]]:
    try:
        _df, _meta = pyreadstat.read_xport(str(path))
        return True, None
    except Exception as e:
        return False, str(e)


def write_json_report(path: Path, payload: Dict[str, Any]) -> None:
    mkdirp(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def download_requests(url: str, dest_path: Path, headers=DEFAULT_HEADERS, timeout: int = 30) -> Tuple[bool, Dict[str, Any]]:
    info: Dict[str, Any] = {"url": url}
    try:
        with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=timeout) as r:
            info["status_code"] = r.status_code
            info["content_type"] = r.headers.get("Content-Type", "")
            length_header = r.headers.get("Content-Length")
            info["content_length_header"] = int(length_header) if length_header and length_header.isdigit() else None

            content = r.content
            if r.status_code != 200:
                if content:
                    safe_write_bytes(dest_path.with_suffix(dest_path.suffix + ".bad.html"), content)
                info["error"] = f"HTTP {r.status_code}"
                return False, info

            if "html" in info["content_type"].lower() or (info["content_length_header"] and info["content_length_header"] < MIN_VALID_SIZE) or len(content) < MIN_VALID_SIZE:
                safe_write_bytes(dest_path.with_suffix(dest_path.suffix + ".bad.html"), content)
                info["error"] = "Response appears to be HTML or too small"
                info["downloaded_bytes"] = len(content)
                return False, info

            safe_write_bytes(dest_path, content)
            info["downloaded_bytes"] = len(content)
            return True, info
    except Exception as e:
        info["error"] = f"requests exception: {e}"
        return False, info


def download_curl(url: str, dest_path: Path, timeout: int = 120) -> Tuple[bool, Dict[str, Any]]:
    info: Dict[str, Any] = {"url": url}
    # include UA and silent fail on non-200
    cmd = ["curl", "-L", "-f", "-sS", "-A", DEFAULT_HEADERS["User-Agent"], "-o", str(dest_path), url]
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
        size = dest_path.stat().st_size if dest_path.exists() else 0
        info["downloaded_bytes"] = size
        if looks_like_html_file(dest_path) or size < MIN_VALID_SIZE:
            with open(dest_path, "rb") as fh:
                content = fh.read()
            safe_write_bytes(dest_path.with_suffix(dest_path.suffix + ".bad.html"), content)
            info["error"] = "curl downloaded HTML or tiny file"
            return False, info
        info["status_code"] = 200
        return True, info
    except subprocess.CalledProcessError as e:
        info["error"] = f"curl returned non-zero: {e}"
        return False, info
    except Exception as e:
        info["error"] = f"curl exception: {e}"
        return False, info


def process_single_file(cycle: str, fname: str, out_root: Path, convert: bool, force: bool) -> Dict[str, Any]:
    report: Dict[str, Any] = {"filename": fname}
    cycle_dir = out_root / cycle
    mkdirp(cycle_dir)
    dest = cycle_dir / fname

    # Build CDC Public DataFiles URL using the start year of the cycle
    year = cycle.split("_", 1)[0]  # '2005_2006' -> '2005'
    # CDC uses lowercase .xpt in that path
    url = f"{NHANES_BASE}/{year}/DataFiles/{fname.replace('.XPT', '.xpt')}"
    report["url"] = url

    if dest.exists() and not force:
        report["existing_size"] = dest.stat().st_size
        is_valid, err = validate_xpt(dest)
        report["valid"] = is_valid
        if is_valid:
            report["note"] = "exists_and_valid"
        else:
            report["note"] = "exists_but_invalid"
            ok, info = download_requests(url, dest)
            report["redownload_attempt"] = info
            if not ok:
                ok2, info2 = download_curl(url, dest)
                report["curl_attempt_on_invalid"] = info2
                ok = ok2
            if ok:
                is_valid2, err2 = validate_xpt(dest)
                report["valid"] = is_valid2
                if is_valid2:
                    report["note"] = "replaced_with_valid_file"
                else:
                    report["note"] = "re-downloaded_but_still_invalid"
                    report["validation_error_after_redownload"] = err2
    else:
        ok, info = download_requests(url, dest)
        report.update(info)
        if not ok:
            ok2, info2 = download_curl(url, dest)
            report["curl_attempt"] = info2
            if ok2:
                report["downloaded_via"] = "curl"
            ok = ok2

        if ok:
            report["downloaded_via"] = report.get("downloaded_via", "requests")
            is_valid, err = validate_xpt(dest)
            report["valid"] = is_valid
            if not is_valid:
                report["validation_error"] = err
        else:
            report["valid"] = False

    if convert and report.get("valid"):
        csv_dir = cycle_dir / "csv"
        mkdirp(csv_dir)
        csv_name = fname.replace(".XPT", ".csv")
        csv_path = csv_dir / csv_name
        try:
            df, meta = pyreadstat.read_xport(str(dest))
            df.to_csv(csv_path, index=False)
            report["converted"] = True
            report["csv_path"] = str(csv_path)
            report["n_rows"] = int(len(df))
            report["n_columns"] = int(len(df.columns))
        except Exception as e:
            report["converted"] = False
            report["convert_error"] = str(e)

    return report


def process_cycle(cycle: str, out_root: Path, convert: bool, force: bool) -> Dict[str, Any]:
    print(f"\nProcessing cycle {cycle}...")
    letter = CYCLE_LETTER.get(cycle)
    if not letter:
        return {"cycle": cycle, "error": "unknown cycle mapping (letter missing)", "files": {}}

    cycle_report: Dict[str, Any] = {"cycle": cycle, "files": {}}
    for base in DATASETS:
        fname = build_filename(base, letter)
        info = process_single_file(cycle, fname, out_root, convert=convert, force=force)
        cycle_report["files"][fname] = info
        if info.get("valid"):
            print(f"  ✓ {fname}: OK")
        else:
            print(f"  ERROR {fname}: {info.get('validation_error') or info.get('error') or info.get('note')}")

    return cycle_report


def main():
    parser = argparse.ArgumentParser(description="Download and validate NHANES XPT files (robust).")
    parser.add_argument("--out", "-o", default="data/raw", help="Output root for raw files")
    parser.add_argument("--report", "-r", default="reports", help="Report folder")
    parser.add_argument("--convert", action="store_true", help="Also convert validated XPT -> CSV under data/raw/<cycle>/csv/")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--cycles", nargs="*", help="Limit which cycles to process (defaults to all)", default=DEFAULT_CYCLES)
    args = parser.parse_args()

    out_root = Path(args.out)
    report_root = Path(args.report)
    mkdirp(out_root)
    mkdirp(report_root)

    aggregate = {"cycles": [], "summary": {"checked": 0, "valid": 0, "invalid": 0}}
    for cycle in args.cycles:
        rep = process_cycle(cycle, out_root, convert=args.convert, force=args.force)
        rep_path = report_root / f"{cycle}_report.json"
        write_json_report(rep_path, rep)
        aggregate["cycles"].append(str(rep_path))
        for fname, info in rep.get("files", {}).items():
            aggregate["summary"]["checked"] += 1
            if info.get("valid"):
                aggregate["summary"]["valid"] += 1
            else:
                aggregate["summary"]["invalid"] += 1

    agg_path = report_root / "aggregate_report.json"
    write_json_report(agg_path, aggregate)
    print("\nAggregate report saved:", str(agg_path))
    print("Done.")


if __name__ == "__main__":
    main()
