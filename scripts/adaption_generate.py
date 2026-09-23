"""Generate a synthetic anneal corpus with Adaption's `invent`, and never spend without quoting.

    python scripts/adaption_generate.py --rows 200 --tag pilot        # quotes, asks, generates
    python scripts/adaption_generate.py --rows 200 --tag pilot --yes  # skip the confirmation
    python scripts/adaption_generate.py --resume <dataset_id>         # re-attach to a running job

Cost is per *row* and flat (0.1 credits/row, measured), regardless of how long a row is. That single
fact shapes the whole design: the budget is 8,500 rows total, so the lever worth pulling is row
*length*, not row count, and `dataset_prompt` therefore asks for substantial multi-paragraph passages
rather than one-line Q&A. A short row and a long row cost the same and only one of them is worth
annealing on.

What this deliberately does *not* do is ask for benchmark-shaped items. Prompting for "PIQA-style
multiple choice about physical commonsense" would invite the generator to reproduce PIQA items it
memorised, which would be training on the test set with extra steps -- and `decontaminate.py` matches
n-grams, so a reworded leak walks straight through it. The prompts below target the *underlying
skill*, everyday physical cause and effect described in prose, and `scripts/adaption_decontam.py`
measures the overlap afterwards rather than assuming it is zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.adaption_probe import call, load_key  # noqa: E402

# Prose, not quiz items. Each mixture targets a capability the eval gap points at, using the domain
# taxonomy the API actually exposes (see runs/adaption_probe.json).
MIXTURES = {
    # PIQA: procedural physical knowledge -- what happens when you do things to objects.
    "physical": {
        "domains": ["how_to"],
        "subdomains": ["how_to.home_diy", "how_to.cooking_food_prep",
                       "how_to.vehicle_maintenance", "how_to.crafts_making",
                       "how_to.outdoor_activities", "how_to.personal_care_beauty"],
        "dataset_prompt": (
            "Write detailed explanatory prose about everyday physical tasks and why they work the "
            "way they do. Each passage should be several paragraphs of continuous text describing a "
            "concrete task with ordinary household objects, materials and tools: what you do, in "
            "what order, and crucially WHY each step is necessary in terms of physical properties "
            "such as weight, friction, heat, moisture, leverage, adhesion, elasticity and balance. "
            "Explain what goes wrong if a step is skipped or done out of order, and why a "
            "plausible-looking alternative approach fails. Prefer mundane situations over unusual "
            "ones. Write flowing explanatory paragraphs. Do not write lists. Do not write questions "
            "and answers. Do not write multiple-choice items."
        ),
    },
    # HellaSwag: what plausibly happens next in ordinary events.
    "everyday": {
        "domains": ["social", "parenting_family", "how_to"],
        "subdomains": None,
        "dataset_prompt": (
            "Write detailed narrative prose describing ordinary everyday situations as they unfold "
            "over time. Each passage should be several paragraphs following a mundane scene from "
            "beginning to end, such as someone cooking a meal, fixing something around the house, "
            "getting ready to go out, or dealing with a small domestic problem. Narrate what "
            "happens next at each point and why that follows naturally from what came before. "
            "Include the physical details of objects being handled and the ordinary consequences of "
            "actions. Write continuous narrative paragraphs. Do not write lists. Do not write "
            "questions and answers."
        ),
    },
}


def quote(key: str, rows: int, mix: dict, name: str) -> dict:
    body = {"rows": rows, "training_type": "instruction_dataset", "estimate": True,
            "name": name, "dataset_prompt": mix["dataset_prompt"], "domains": mix["domains"]}
    if mix.get("subdomains"):
        body["subdomains"] = mix["subdomains"]
    status, res = call("/api/v1/datasets/invent", key, body)
    if status != 200:
        sys.exit(f"quote failed: HTTP {status} {json.dumps(res)[:400]}")
    return res


def launch(key: str, rows: int, mix: dict, name: str) -> dict:
    body = {"rows": rows, "training_type": "instruction_dataset", "estimate": False,
            "name": name, "dataset_prompt": mix["dataset_prompt"], "domains": mix["domains"],
            # An idempotency key means a retry after a dropped connection re-attaches to the same
            # job instead of paying for a second one.
            "idempotency_key": f"kestrel-{name}-{rows}"}
    if mix.get("subdomains"):
        body["subdomains"] = mix["subdomains"]
    status, res = call("/api/v1/datasets/invent", key, body)
    if status != 200:
        sys.exit(f"launch failed: HTTP {status} {json.dumps(res)[:400]}")
    return res


def poll(key: str, ds_id: str, timeout_min: int = 90) -> dict:
    """Wait for the job, printing progress. Generation is minutes, not seconds."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_min * 60:
        status, res = call(f"/api/v1/datasets/{ds_id}/status", key)
        if status != 200:
            print(f"  status HTTP {status}: {json.dumps(res)[:200]}")
            time.sleep(15)
            continue
        st = res.get("status")
        prog = res.get("progress") or {}
        line = (f"  [{int(time.time()-t0):>4}s] {st} "
                f"{prog.get('processed_rows','?')}/{prog.get('total_rows','?')}")
        if line != last:
            print(line, flush=True)
            last = line
        if st == "succeeded":
            return res
        if st in ("failed", "awaiting_input"):
            print(json.dumps(res, indent=2)[:1000])
            sys.exit(f"job ended: {st}")
        time.sleep(15)
    sys.exit("timed out waiting for generation")


def download(key: str, ds_id: str, out: Path) -> dict:
    status, res = call(f"/api/v1/datasets/{ds_id}/download", key, timeout=300)
    if status != 200:
        sys.exit(f"download failed: HTTP {status} {json.dumps(res)[:400]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def row_count(data) -> int:
    if isinstance(data, list):
        return len(data)
    for k in ("rows", "data", "records", "items"):
        v = data.get(k)
        if isinstance(v, list):
            return len(v)
    return 0


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--mixture", default="physical", choices=sorted(MIXTURES))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    ap.add_argument("--resume", default=None, help="dataset id of an already-launched job")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synthetic")
    args = ap.parse_args()

    key = load_key()
    name = f"kestrel-{args.mixture}-{args.tag}"
    out = args.out_dir / f"{name}.json"

    if args.resume:
        ds_id = args.resume
        print(f"re-attaching to {ds_id}")
    else:
        mix = MIXTURES[args.mixture]
        q = quote(key, args.rows, mix, name)
        est, avail = q.get("estimated_credits"), q.get("available_credits")
        left = (avail - est) if isinstance(est, (int, float)) and isinstance(avail, (int, float)) else "?"
        print(f"mixture   : {args.mixture}  ({', '.join(mix['domains'])})")
        print(f"rows      : {args.rows}")
        print(f"quote     : {est} credits of {avail} available  -> {left} left after")
        if not args.yes:
            if input("proceed and spend? [y/N] ").strip().lower() != "y":
                sys.exit("aborted; nothing spent")
        res = launch(key, args.rows, mix, name)
        # The invent response names it `id`; every other endpoint names it `dataset_id`, and the
        # list endpoint only returns the latter. Accept both rather than depend on which one a
        # given deployment sends.
        ds_id = res.get("dataset_id") or res.get("id")
        if not ds_id:
            sys.exit(f"launched but no dataset id in response: {json.dumps(res)[:400]}")
        print(f"launched  : {ds_id}  status={res.get('status')}")

    poll(key, ds_id)
    data = download(key, ds_id, out)
    n = row_count(data)
    print(f"\ndownloaded {n} rows -> {out}")

    meta = {"dataset_id": ds_id, "mixture": args.mixture, "rows_requested": args.rows,
            "name": name, "downloaded_rows": n, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (args.out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
