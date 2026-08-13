from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class StructuralPair:
    a: str
    b: str
    expected_sign: str   # "positive" or "negative"
    rationale: str


# Pairs with strong economic priors — breaks here carry more signal than random pair breaks.
STRUCTURAL_PAIRS: List[StructuralPair] = [
    StructuralPair("AUDUSD=X", "CPER",     "positive", "AUD as China/commodity proxy ↔ copper"),
    StructuralPair("CAD=X",    "USO",      "negative", "USDCAD inverse to oil (CAD strengthens with oil)"),
    StructuralPair("JPY=X",    "TLT",      "negative", "USDJPY rises when long UST sells off (yield diff)"),
    StructuralPair("GLD",      "DX-Y.NYB", "negative", "Gold inverse to USD"),
    StructuralPair("EEM",      "DX-Y.NYB", "negative", "EM equities inverse to USD"),
    StructuralPair("GLD",      "TLT",      "positive", "Gold and long bonds both real-yield sensitive"),
    StructuralPair("HYG",      "SPY",      "positive", "High-yield credit ↔ equity beta"),
    StructuralPair("FXI",      "CPER",     "positive", "China equity ↔ copper"),
    StructuralPair("EURUSD=X", "GBPUSD=X", "positive", "EUR-GBP co-move vs USD"),
    StructuralPair("SPY",      "TLT",      "negative", "Stock-bond inverse (regime-dependent)"),
    StructuralPair("QQQ",      "TLT",      "negative", "Long-duration tech ↔ long bonds (post-2020 era)"),
    StructuralPair("SLV",      "GLD",      "positive", "Silver follows gold"),
    StructuralPair("EFA",      "SPY",      "positive", "Global developed equity beta"),
    StructuralPair("IWM",      "SPY",      "positive", "Small-cap equity beta"),
    StructuralPair("LQD",      "TLT",      "positive", "IG credit duration-driven"),
]


def all_pairs() -> List[tuple[str, str]]:
    return [(p.a, p.b) for p in STRUCTURAL_PAIRS]


def expected_sign(a: str, b: str) -> str | None:
    for p in STRUCTURAL_PAIRS:
        if {p.a, p.b} == {a, b}:
            return p.expected_sign
    return None
