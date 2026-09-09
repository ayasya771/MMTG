"""Generate the full synthetic multimodal dataset.

Simulates the macro economy, then renders every modality from it: charts
with structural break captions, monthly central bank statements and news
with ground truth triplets, and port scenes with container density labels.

Usage:
    python scripts/make_synthetic_data.py [--months 480] [--seed 7]
                                          [--chart-stride 1] [--ports-per-month 3]
"""

import argparse

import _bootstrap

from mmtg.config import ChartConfig, Paths, PortConfig, SimConfig
from mmtg.data.charts import build_chart_dataset
from mmtg.data.macro_sim import simulate
from mmtg.data.ports import build_port_dataset
from mmtg.data.statements import generate_corpus
from mmtg.utils import Timer, log, save_json, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", default="generated",
                        help="dataset subdirectory under data/")
    parser.add_argument("--months", type=int, default=SimConfig().months)
    parser.add_argument("--seed", type=int, default=SimConfig().seed)
    parser.add_argument("--chart-stride", type=int, default=1)
    parser.add_argument("--ports-per-month", type=int, default=3)
    args = parser.parse_args()

    paths = Paths(data_name=args.data_name).ensure()

    with Timer() as t_sim:
        panel = simulate(SimConfig(months=args.months, seed=args.seed))
        save_json(panel.to_dict(), f"{paths.data_dir}/panel.json")
    log(f"simulated {panel.n_months} months ({t_sim})")

    with Timer() as t_corp:
        corpus = generate_corpus(panel, seed=args.seed + 22)
        save_jsonl(corpus, f"{paths.data_dir}/corpus.jsonl")
    n_trip = sum(len(d["triplets"]) for d in corpus)
    log(f"wrote {len(corpus)} statements, {n_trip} ground truth triplets ({t_corp})")

    with Timer() as t_charts:
        chart_cfg = ChartConfig(stride=args.chart_stride)
        chart_rows = build_chart_dataset(panel, chart_cfg, paths.charts_dir)
        save_jsonl(chart_rows, f"{paths.data_dir}/chart_manifest.jsonl")
    log(f"rendered {len(chart_rows)} charts ({t_charts})")

    with Timer() as t_ports:
        port_cfg = PortConfig(per_month=args.ports_per_month)
        port_rows = build_port_dataset(panel, port_cfg, paths.ports_dir)
        save_jsonl(port_rows, f"{paths.data_dir}/port_manifest.jsonl")
    log(f"rendered {len(port_rows)} port scenes ({t_ports})")


if __name__ == "__main__":
    main()
