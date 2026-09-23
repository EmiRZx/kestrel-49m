"""Probe the Adaption API for what a credit actually buys, without spending one.

    python scripts/adaption_probe.py                 # domains + cost curve, spends nothing
    python scripts/adaption_probe.py --rows 2000     # quote one specific size

Everything here is read-only or `estimate=True`. The invent endpoint quotes a job -- returning
`estimated_credits` and `available_credits` -- without queueing it, so the whole cost model can be
mapped for free. That matters because the credit-to-row rate is the number the entire synthetic-data
plan is sized against, and the last measurement (0.1 credits/row against a 50-credit balance) was
taken before the account was upgraded.

The key is read from the environment or `.env` and never printed. Only its length and prefix are
ever shown, which is enough to tell "loaded the right key" from "loaded nothing".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://api.prod.adaptionlabs.ai"


def load_key() -> str:
    """Environment first, then `.env`. The file is gitignored; the value is never logged."""
    key = os.environ.get("ADAPTION_API_KEY", "").strip()
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("ADAPTION_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        sys.exit("ADAPTION_API_KEY not found in environment or .env")
    return key


def call(path: str, key: str, payload: dict | None = None, timeout: int = 60) -> tuple[int, dict]:
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body or "{}")
        except json.JSONDecodeError:
            return e.code, {"raw": body[:500]}
    except Exception as e:                                    # noqa: BLE001
        return 0, {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*",
                    default=[1, 100, 1000, 10000, 100000],
                    help="row counts to quote; every call uses estimate=True and spends nothing")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "adaption_probe.json")
    args = ap.parse_args()

    key = load_key()
    print(f"key loaded: {len(key)} chars, prefix {key[:3]}***\n")

    report: dict = {"base": BASE, "quotes": []}

    # 1. Domains -- static, free, and tells us which taxonomy buckets exist. The mixture design
    #    depends on whether there is a physical/commonsense bucket at all.
    status, doms = call("/api/v1/datasets/invent/domains", key)
    print(f"GET  /datasets/invent/domains -> {status}")
    if status == 200:
        ds = doms.get("domains", [])
        report["domains"] = ds
        print(f"  {len(ds)} domains")
        for d in ds:
            subs = d.get("subdomains", [])
            print(f"   - {d.get('code','?'):<28} {d.get('title','')}  ({len(subs)} subdomains)")
    else:
        print(f"  {json.dumps(doms)[:400]}")
        report["domains_error"] = doms

    # 2. Quote a cost curve. If credits/row is constant the plan is a simple division; if it has a
    #    floor or breaks, the pilot size has to be chosen around that.
    print("\nPOST /datasets/invent  (estimate=True -- quotes only, queues nothing)")
    for n in args.rows:
        status, res = call("/api/v1/datasets/invent", key, {
            "rows": n,
            "training_type": "instruction_dataset",
            "estimate": True,
            "dataset_prompt": "Short passages describing everyday physical cause and effect.",
        })
        if status == 200:
            est, avail = res.get("estimated_credits"), res.get("available_credits")
            per = (est / n) if isinstance(est, (int, float)) and n else None
            line = f"  rows={n:<8} est={est!s:<12} avail={avail!s:<12}"
            if per is not None:
                line += f" per_row={per:.6f}"
            print(line)
            report["quotes"].append({"rows": n, "estimated_credits": est,
                                     "available_credits": avail, "per_row": per})
        else:
            print(f"  rows={n:<8} HTTP {status}: {json.dumps(res)[:300]}")
            report["quotes"].append({"rows": n, "status": status, "error": res})
            if status in (401, 403):
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
