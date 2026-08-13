from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str


UNIVERSE: List[Asset] = [
    # Equities (regional ETFs, NYSE close-aligned)
    Asset("SPY",  "S&P 500",            "equity"),
    Asset("QQQ",  "Nasdaq 100",         "equity"),
    Asset("IWM",  "Russell 2000",       "equity"),
    Asset("EFA",  "DM ex-US",           "equity"),
    Asset("EEM",  "Emerging Markets",   "equity"),
    Asset("FXI",  "China large-cap",    "equity"),

    # FX (yfinance forex pairs)
    Asset("EURUSD=X", "EUR/USD",        "fx"),
    Asset("GBPUSD=X", "GBP/USD",        "fx"),
    Asset("JPY=X",    "USD/JPY",        "fx"),
    Asset("AUDUSD=X", "AUD/USD",        "fx"),
    Asset("CAD=X",    "USD/CAD",        "fx"),
    Asset("CHF=X",    "USD/CHF",        "fx"),
    Asset("DX-Y.NYB", "DXY",            "fx"),

    # Commodities (ETFs for calendar alignment)
    Asset("GLD",  "Gold",               "commodity"),
    Asset("SLV",  "Silver",             "commodity"),
    Asset("USO",  "WTI Crude",          "commodity"),
    Asset("UNG",  "Natural Gas",        "commodity"),
    Asset("CPER", "Copper",             "commodity"),
    Asset("DBA",  "Agriculture",        "commodity"),

    # Rates / credit (ETF proxies)
    Asset("TLT",  "20+y UST",           "rates"),
    Asset("IEF",  "7-10y UST",          "rates"),
    Asset("SHY",  "1-3y UST",           "rates"),
    Asset("LQD",  "IG Credit",          "rates"),
    Asset("HYG",  "HY Credit",          "rates"),
]

# VIX is used for regime classification, not as a tradable in the universe
REGIME_TICKER = "^VIX"

# Common factors used to residualize partial correlations
COMMON_FACTORS = ["DX-Y.NYB", "SPY", "TLT", "USO"]


def tickers() -> List[str]:
    return [a.ticker for a in UNIVERSE]


def all_tickers() -> List[str]:
    return tickers() + [REGIME_TICKER]


def asset_class_map() -> dict:
    return {a.ticker: a.asset_class for a in UNIVERSE}
