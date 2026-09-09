"""Canonical schema of the Dynamic Knowledge Graph.

Entities and relations are defined once here and shared by the synthetic
corpus generator (which writes text using these surface forms) and by the
information extraction engine (which parses text back into canonical
triplets). Keeping the two sides on one schema makes extraction quality
measurable: the corpus carries ground truth triplets and the test suite
asserts the extractor recovers them.
"""

from __future__ import annotations

ENTITIES: dict[str, dict] = {
    "Federal_Reserve": {
        "type": "institution",
        "surfaces": ["the federal reserve", "the federal open market committee",
                     "the fed", "the committee", "the fomc", "policymakers"],
        "gloss": "the united states central bank setting monetary policy",
    },
    "Policy_Rate": {
        "type": "indicator",
        "surfaces": ["the policy rate", "the federal funds rate",
                     "the target range", "rate hikes", "higher policy rates",
                     "interest rates", "rates"],
        "gloss": "the federal funds policy interest rate",
    },
    "CPI_Inflation": {
        "type": "indicator",
        "surfaces": ["headline inflation", "consumer prices", "core inflation",
                     "price pressures", "cpi", "inflation"],
        "gloss": "consumer price inflation and underlying price pressures",
    },
    "GDP_Growth": {
        "type": "indicator",
        "surfaces": ["economic growth", "gdp growth", "improving activity",
                     "resilient demand", "output", "activity", "the economy"],
        "gloss": "real gdp growth and aggregate economic activity",
    },
    "Unemployment": {
        "type": "indicator",
        "surfaces": ["unemployment", "the labor market", "payroll growth",
                     "joblessness"],
        "gloss": "unemployment and labor market conditions",
    },
    "Real_Yields": {
        "type": "indicator",
        "surfaces": ["real yields", "inflation adjusted yields"],
        "gloss": "inflation adjusted treasury yields",
    },
    "Long_End_Yields": {
        "type": "indicator",
        "surfaces": ["long end yields", "the ten year yield",
                     "long term treasury yields"],
        "gloss": "yields on long maturity treasury bonds",
    },
    "Yield_Curve": {
        "type": "market_factor",
        "surfaces": ["the yield curve", "the 2s10s curve", "the curve"],
        "gloss": "the slope of the treasury yield curve",
    },
    "Tech_Multiples": {
        "type": "market_factor",
        "surfaces": ["technology valuation multiples", "tech multiples",
                     "growth stock valuations", "equity duration"],
        "gloss": "valuation multiples of long duration technology equities",
    },
    "Financial_Conditions": {
        "type": "market_factor",
        "surfaces": ["financial conditions", "credit conditions"],
        "gloss": "broad financial and credit conditions",
    },
    "Policy_Tightening": {
        "type": "theme",
        "surfaces": ["additional policy firming", "further tightening",
                     "aggressive tightening", "a restrictive stance",
                     "policy firming"],
        "gloss": "the theme of tighter restrictive monetary policy",
    },
    "Policy_Easing": {
        "type": "theme",
        "surfaces": ["prospects of rate cuts", "policy accommodation",
                     "an easier policy stance", "rate cuts"],
        "gloss": "the theme of easier accommodative monetary policy",
    },
    "Recession_Risk": {
        "type": "theme",
        "surfaces": ["recession fears", "recession risk", "a hard landing",
                     "hard landing risks"],
        "gloss": "the risk of an economic recession",
    },
    "Growth_Slowdown": {
        "type": "theme",
        "surfaces": ["slowing activity", "the growth slowdown",
                     "the slowdown in output", "softening demand"],
        "gloss": "a broad slowdown in economic activity",
    },
    "Port_Congestion": {
        "type": "theme",
        "surfaces": ["port congestion", "shipping backlogs",
                     "container backlogs at major ports",
                     "vessel queues at long beach",
                     "vessel queues at shanghai", "easing port congestion"],
        "gloss": "congestion and container backlogs at global shipping ports",
    },
    "Supply_Chains": {
        "type": "theme",
        "surfaces": ["supply chains", "global supply chains",
                     "supply disruptions", "logistics strains"],
        "gloss": "the state of global supply chains and logistics",
    },
    "US_Equities": {
        "type": "asset", "ticker": "SPY",
        "surfaces": ["the s&p 500", "us equities", "the broad equity market",
                     "stocks"],
        "gloss": "broad united states equities tracked by spy",
    },
    "Tech_Equities": {
        "type": "asset", "ticker": "QQQ",
        "surfaces": ["the nasdaq 100", "technology shares", "megacap tech",
                     "growth equities"],
        "gloss": "technology heavy nasdaq equities tracked by qqq",
    },
    "Long_Treasuries": {
        "type": "asset", "ticker": "TLT",
        "surfaces": ["long treasuries", "long duration bonds", "the long bond",
                     "treasury bonds"],
        "gloss": "long maturity treasury bonds tracked by tlt",
    },
    "Gold": {
        "type": "asset", "ticker": "GLD",
        "surfaces": ["gold", "bullion"],
        "gloss": "gold bullion tracked by gld",
    },
    "US_Dollar": {
        "type": "asset", "ticker": "DXY",
        "surfaces": ["the us dollar", "the dollar", "the greenback"],
        "gloss": "the trade weighted united states dollar index",
    },
    "Crude_Oil": {
        "type": "asset", "ticker": "USO",
        "surfaces": ["crude oil", "energy prices", "wti", "oil"],
        "gloss": "crude oil prices tracked by uso",
    },
}

RELATIONS: dict[str, dict] = {
    "raises":    {"surfaces": ["raised", "hiked", "increased"], "direction": "up"},
    "cuts":      {"surfaces": ["cut", "lowered", "reduced"], "direction": "down"},
    "holds":     {"surfaces": ["held", "maintained"], "direction": "flat"},
    "lifts":     {"surfaces": ["pushed up", "drove up", "pushed higher"], "direction": "up"},
    "boosts":    {"surfaces": ["boosted", "supported", "buoyed", "lifted", "burnished"], "direction": "up"},
    "compresses": {"surfaces": ["compressed", "squeezed"], "direction": "down"},
    "weighs_on": {"surfaces": ["weighed on", "dragged on", "dragged lower", "hurt"], "direction": "down"},
    "stokes":    {"surfaces": ["stoked", "fueled", "amplified", "rekindled"], "direction": "up"},
    "cools":     {"surfaces": ["cooled", "restrained", "curbed", "relieved pressure on"], "direction": "down"},
    "inverts":   {"surfaces": ["inverted"], "direction": "down"},
    "steepens":  {"surfaces": ["steepened"], "direction": "up"},
    "signals":   {"surfaces": ["signaled", "flagged", "telegraphed"], "direction": "flat"},
}

HAWKISH_TERMS = [
    "elevated", "unacceptably", "persistent", "restrictive", "firming",
    "tightening", "vigilant", "upside risks", "raised", "hiked", "overheating",
    "entrenched",
]
DOVISH_TERMS = [
    "accommodative", "downside risks", "patient", "cut", "lowered", "easing",
    "soft landing", "cooling", "moderated", "accommodation", "disinflation",
]


def entity_ticker(entity: str) -> str | None:
    return ENTITIES.get(entity, {}).get("ticker")


def all_surface_pairs() -> list[tuple[str, str]]:
    """(surface, canonical) pairs sorted longest surface first."""
    pairs = [(s, name) for name, spec in ENTITIES.items() for s in spec["surfaces"]]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def all_relation_pairs() -> list[tuple[str, str]]:
    pairs = [(s, name) for name, spec in RELATIONS.items() for s in spec["surfaces"]]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)
