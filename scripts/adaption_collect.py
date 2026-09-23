"""Collect finished Adaption jobs, then exit. Safe to run whenever the machine happens to be on.

    python scripts/adaption_collect.py                 # check every known job once, download what is done
    python scripts/adaption_collect.py --watch         # keep checking until everything is collected

Generation runs on Adaption's servers, not here. The pilot's own timestamps prove it: 1,636 seconds
between `created_at` and `updated_at`, during which this machine did nothing but send one HTTP
request every fifteen seconds. So there is no reason for a laptop to sit awake through a long job --
the work continues whether anything is watching or not, and a finished dataset stays on the server
until it is fetched.

That makes the right shape for a long job "launch, walk away, collect later" rather than one process
babysitting a socket for hours. `--watch` exists for when the machine is going to be on anyway; the
default single pass is the one to use otherwise, and re-running it is free because downloading a
dataset costs nothing. Nothing here can spend credits: it only ever reads.

A job that is still running is reported and skipped, not waited on. A job already downloaded is
skipped too, so this is safe to run in a loop, from a scheduled task, or by hand whenever.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.adaption_probe import BASE, call, load_key  # noqa: E402

OUT_DIR = ROOT / "data" / "synthetic"


def list_datasets(key: str) -> list[dict]:
    status, res = call("/api/v1/datasets", key)
    if status != 200:
        sys.exit(f"list failed: HTTP {status} {json.dumps(res)[:300]}")
    rows = res if isinstance(res, list) else (res.get("datasets") or res.get("data") or [])
    return [d for d in rows if str(d.get("name", "")).startswith("kestrel")]


def fetch(key: str, ds_id: str, out: Path) -> int:
    """Download one finished dataset. The endpoint answers text/csv."""
    req = urllib.request.Request(f"{BASE}/api/v1/datasets/{ds_id}/download")
    req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read().decode("utf-8", "replace")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Raw bytes first: a parsing bug must never cost a re-generation.
    out.with_suffix(".csv").write_text(raw, encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return len(rows)


def sweep(key: str) -> tuple[int, int]:
    """One pass. Returns (collected, still_running)."""
    collected = running = 0
    for d in list_datasets(key):
        name, ds_id, st = d.get("name"), d.get("dataset_id"), d.get("status")
        out = OUT_DIR / f"{name}.json"
        if out.exists():
            continue
        if st == "succeeded":
            n = fetch(key, ds_id, out)
            print(f"  collected {name}: {n:,} rows -> {out}", flush=True)
            (OUT_DIR / f"{name}.meta.json").write_text(json.dumps({
                "dataset_id": ds_id, "name": name, "rows": n,
                "created_at": d.get("created_at"), "updated_at": d.get("updated_at"),
                "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2), encoding="utf-8")
            collected += 1
        elif st in ("running", "pending"):
            print(f"  {st:<9} {name}  (since {d.get('created_at')})", flush=True)
            running += 1
        else:
            print(f"  {st:<9} {name}  -- not collectable", flush=True)
    return collected, running


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="keep polling until nothing is running")
    ap.add_argument("--every", type=int, default=120, help="seconds between passes when watching")
    args = ap.parse_args()

    key = load_key()
    while True:
        print(f"[{time.strftime('%H:%M:%S')}] checking")
        collected, running = sweep(key)
        if not args.watch or running == 0:
            print(f"\n{collected} collected, {running} still running.")
            if running:
                print("Nothing is waiting on this machine -- generation continues on their servers.\n"
                      "Re-run this script any time to pick the results up.")
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
