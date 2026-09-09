"""Download the real free data sources (needs network access).

Pulls FRED macro series, monthly asset prices from Stooq, and FOMC statement
texts from federalreserve.gov into data/raw/. Requires outbound network
access, so run it on a machine that has it. Satellite
imagery (EuroSAT, SpaceNet, Sentinel-2 port chips through Google Earth
Engine) is a one time manual download; see README section "Real data".

Usage:
    python scripts/download_data.py [--start 1990-01] [--fomc-years 2015 2016 ...]
"""

import argparse
import os

import _bootstrap

from mmtg.config import Paths
from mmtg.data.downloaders import (build_real_panel, fetch_fomc_statements)
from mmtg.utils import log, save_json, save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", default="generated",
                        help="dataset subdirectory under data/")
    parser.add_argument("--start", default="1990-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--fred-api-key", default=None,
                        help="optional, otherwise the public CSV endpoint is used")
    parser.add_argument("--fomc-years", nargs="*", type=int,
                        default=list(range(2006, 2026)))
    args = parser.parse_args()

    paths = Paths(data_name=args.data_name).ensure()

    log("downloading FRED series and Stooq prices")
    panel = build_real_panel(args.start, args.end, args.fred_api_key)
    save_json(panel, os.path.join(paths.raw_dir, "real_panel.json"))
    log(f"real panel: {len(panel['months'])} months "
        f"({panel['months'][0]} to {panel['months'][-1]})")

    log(f"scraping FOMC statements for {len(args.fomc_years)} years")
    statements = fetch_fomc_statements(args.fomc_years)
    save_jsonl(statements, os.path.join(paths.raw_dir, "fomc_statements.jsonl"))
    log(f"saved {len(statements)} statements to data/raw/fomc_statements.jsonl")

    log("done. Charts for real data: render FRED windows with "
        "mmtg.data.charts.render_chart on the real panel, then rerun the "
        "three training phases against data/raw/real_panel.json.")


if __name__ == "__main__":
    main()
