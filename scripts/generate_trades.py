"""Generate a JSON portfolio action from multimodal inputs.

Two modes:
- dataset mode: --month YYYY-MM picks that month's charts, statement and
  port scene from the generated dataset,
- custom mode: point --charts, --statement (text or a file path) and
  optionally --port at your own inputs with --as-of for graph time travel.

Usage:
    python scripts/generate_trades.py --month 2025-06
    python scripts/generate_trades.py --charts my_chart.png \
        --statement statement.txt --as-of 2025-06
"""

import argparse
import json
import os

import _bootstrap

from mmtg.config import Paths
from mmtg.pipeline import MacroTradePipeline
from mmtg.utils import load_jsonl, log, save_json


def dataset_inputs(paths: Paths, month: str) -> tuple[list[str], str, str | None]:
    chart_rows = load_jsonl(f"{paths.data_dir}/chart_manifest.jsonl")
    charts = [os.path.join(paths.charts_dir, r["file"])
              for r in chart_rows if r["month"] == month]
    if not charts:
        available = sorted({r["month"] for r in chart_rows})
        raise SystemExit(f"no charts for {month}; months span "
                         f"{available[0]} to {available[-1]}")
    corpus = load_jsonl(f"{paths.data_dir}/corpus.jsonl")
    statement = next((d["statement"] for d in corpus if d["month"] == month), None)
    if statement is None:
        raise SystemExit(f"no statement for {month}")
    port_rows = load_jsonl(f"{paths.data_dir}/port_manifest.jsonl")
    port = next((os.path.join(paths.ports_dir, r["file"])
                 for r in port_rows if r["month"] == month), None)
    return charts, statement, port


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", default="generated",
                        help="dataset subdirectory under data/")
    parser.add_argument("--run", default="runs/main")
    parser.add_argument("--month", help="dataset mode: generate for this month")
    parser.add_argument("--charts", nargs="+", help="custom chart PNG paths")
    parser.add_argument("--statement", help="custom statement text or file path")
    parser.add_argument("--port", help="custom port image PNG")
    parser.add_argument("--as-of", dest="as_of",
                        help="graph knowledge cutoff month for custom mode")
    parser.add_argument("--out", help="write the action JSON here")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    paths = Paths(data_name=args.data_name)
    run_dir = os.path.join(paths.root, args.run)
    pipe = MacroTradePipeline.load(run_dir, device=args.device)

    if args.month:
        charts, statement, port = dataset_inputs(paths, args.month)
        as_of = args.month
    else:
        if not (args.charts and args.statement and args.as_of):
            raise SystemExit("custom mode needs --charts, --statement and --as-of")
        charts = args.charts
        statement = args.statement
        if os.path.exists(statement):
            with open(statement, encoding="utf-8") as fh:
                statement = fh.read()
        port = args.port
        as_of = args.as_of

    action = pipe.generate(charts, statement, as_of, port_path=port)
    print(json.dumps(action, indent=2))

    out = args.out or os.path.join(run_dir, "actions", f"action_{as_of}.json")
    save_json(action, out)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
