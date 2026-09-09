import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mmtg.config import ChartConfig, PortConfig, SimConfig
from mmtg.data.charts import build_chart_dataset
from mmtg.data.macro_sim import simulate
from mmtg.data.ports import build_port_dataset
from mmtg.data.statements import generate_corpus
from mmtg.tokenizer import FinancialTokenizer


@pytest.fixture(scope="session")
def panel():
    return simulate(SimConfig(months=90, seed=3))


@pytest.fixture(scope="session")
def corpus(panel):
    return generate_corpus(panel, seed=5)


@pytest.fixture(scope="session")
def chart_cfg():
    return ChartConfig(stride=8)


@pytest.fixture(scope="session")
def chart_data(panel, chart_cfg, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("charts"))
    rows = build_chart_dataset(panel, chart_cfg, out)
    return rows, out


@pytest.fixture(scope="session")
def port_data(panel, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("ports"))
    rows = build_port_dataset(panel, PortConfig(per_month=1, seed=9), out)
    return rows, out


@pytest.fixture(scope="session")
def tokenizer(corpus, chart_data):
    rows, _ = chart_data
    texts = [r["caption"] for r in rows]
    texts += [d["statement"] for d in corpus]
    for d in corpus:
        texts.extend(d["news"])
    return FinancialTokenizer.build(texts, vocab_size=2048)
