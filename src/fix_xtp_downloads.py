#!/usr/bin/env python3
"""
fix_xpt_downloads.py
Detect broken .XPT files saved as HTML and re-download them robustly then validate with pyreadstat.
Usage: python scripts/fix_xpt_downloads.py
"""

import json
import glob
import os
import sys
import subprocess
from pathlib import Path

import requests
import pyreadstat

MIN_VALID_SIZE = 2048  # bytes: heuristic min size for a real XPT (change if needed)


def looks_like_html(path):
    try:
        with open(path, 'rb') as fh:
            start = fh.read(1024).lower()
            if b'<html' in start or b'<!doctype' in start or b'<head' in start:
                return True
    except Exception:
        return False
    return False


def valid_xpt(path):
    # Try reading with pyreadstat
    try:
        _df, _meta = pyreadstat.read_xport(str(path))
        return True
    except Exception:
        return False


def download_requests(url, dest):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    print(f"  -> requests GET {url}")
    r = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30)
    # basic sanity checks
    ct = r.headers.get("Content-Type", "").lower()
    length = int(r.headers.get("Content-Length", 0) or 0)
    if 'html' in ct or r.status_code != 200 or (length and length < MIN_VALID_SIZE):
        # still write the content for debugging but mark as suspect
        with open(dest + ".downloaded.html", "wb") as fh:
            fh.write(r.content)
        print(f"     -> Response suspicious: status={r.status_code} content-type={ct} length={length}")
        return False
    # write binary
    with open(dest, "wb") as fh:
        for chunk in r.iter_content(1024 * 16):
            if chunk:
                fh.write(chunk)
    return True


def download_curl(url, dest):
    print(f"  -> curl -L {url}")
    cmd = ["curl", "-L", "-f", "-sS", "-o", dest, url]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        return True
    except subprocess.CalledProcessError as e:
        print("     -> curl failed:", e)
        return False
    except Exception as e:
        print("     -> curl exception:", e)
        return False


def main():
    reports = sorted(glob.glob("reports/*_report.json"))
    if not reports:
        print("No reports found in reports/*.json. Exiting.")
        sys.exit(1)

    summary = {"checked": 0, "fixed": 0, "failed": 0}
    for rep_file in reports:
        with open(rep_file, "r", encoding="utf-8") as fh:
            rep = json.load(fh)
        cycle = rep.get("cycle") or Path(rep_file).stem.replace("_report", "")
        # NHANES URL cycles use hyphen: 2005-2006 etc.
        cycle_url_fragment = cycle.replace("_", "-")

        files_info = rep.get("files") or {}
        for fname, info in files_info.items():
            summary["checked"] += 1
            local_path = Path("data/raw") / cycle / fname
            local_path_parent = local_path.parent
            local_path_parent.mkdir(parents=True, exist_ok=True)

            need_fix = False
            if not local_path.exists():
                print(f"[MISSING] {cycle}/{fname}")
                need_fix = True
            else:
                size = local_path.stat().st_size
                if size < MIN_VALID_SIZE:
                    print(f"[TOO SMALL] {cycle}/{fname} size={size}")
                    need_fix = True
                elif looks_like_html(local_path):
                    print(f"[HTML] {cycle}/{fname} appears to be HTML")
                    need_fix = True
                else:
                    # try validation with pyreadstat
                    try:
                        if not valid_xpt(local_path):
                            print(f"[INVALID] {cycle}/{fname} failed pyreadstat validation")
                            need_fix = True
                    except Exception as e:
                        print(f"[CHECK ERROR] {cycle}/{fname} pyreadstat exception: {e}")
                        need_fix = True

            if not need_fix:
                print(f"[OK] {cycle}/{fname}")
                continue

            # get URL from report if present, else construct canonical NHANES URL
            url = info.get("url") or info.get("download_url") or info.get("source") or None
            if not url:
                url = f"https://wwwn.cdc.gov/Nchs/Nhanes/{cycle_url_fragment}/{fname}"

            # Try requests download
            try:
                ok = download_requests(url, str(local_path))
                if ok and valid_xpt(local_path):
                    print(f"[FIXED] {cycle}/{fname} (requests)")
                    summary["fixed"] += 1
                    continue
            except Exception as ex:
                print("   requests download/validate exception:", ex)

            # Fallback to curl
            try:
                ok2 = download_curl(url, str(local_path))
                if ok2 and valid_xpt(local_path):
                    print(f"[FIXED] {cycle}/{fname} (curl)")
                    summary["fixed"] += 1
                    continue
                else:
                    print(f"[FAILED] {cycle}/{fname} could not be validated after download")
                    summary["failed"] += 1
            except Exception as ex:
                print("   curl fallback exception:", ex)
                summary["failed"] += 1

    print("\nSUMMARY:", summary)
    print("If any failed, inspect the *.downloaded.html files in the cycle folder for server error content (login page, 403/404 HTML).")
    print("After all fixed, run your convert step or: python src/check_nhanes_downloads.py --out data/raw --report reports --convert")

if __name__ == "__main__":
    main()
