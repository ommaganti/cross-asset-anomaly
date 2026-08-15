from __future__ import annotations

import pandas as pd


# Curated list of macro/market events for explainability overlay.
# Extend by appending (YYYY-MM-DD, label) tuples or wiring to an econ calendar API.
KNOWN_EVENTS: list[tuple[str, str]] = [
    ("2015-08-11", "PBoC CNY devaluation"),
    ("2015-12-16", "Fed first hike post-GFC"),
    ("2016-06-23", "Brexit vote"),
    ("2016-11-08", "US election (Trump)"),
    ("2018-02-05", "Volmageddon / XIV blowup"),
    ("2018-12-24", "Q4 2018 selloff trough"),
    ("2019-08-14", "UST 2s10s inversion"),
    ("2020-03-09", "Saudi-Russia oil shock"),
    ("2020-03-15", "Fed emergency cut to 0%"),
    ("2020-03-23", "Fed unlimited QE"),
    ("2021-01-27", "Meme-stock / GameStop squeeze"),
    ("2022-02-24", "Russia invades Ukraine"),
    ("2022-03-16", "Fed begins hiking cycle"),
    ("2022-06-15", "Fed +75 bps"),
    ("2022-09-23", "UK gilt / LDI crisis"),
    ("2023-03-10", "SVB collapse"),
    ("2023-03-19", "Credit Suisse / UBS deal"),
    ("2023-10-07", "Israel-Hamas war begins"),
    ("2024-08-05", "Yen carry unwind / VIX spike to ~65"),
    ("2024-09-18", "Fed -50 bps (first cut)"),
    ("2025-04-02", "US 'Liberation Day' tariffs"),
    ("2025-04-09", "Tariff pause / risk reversal"),
]


def event_calendar() -> pd.DataFrame:
    df = pd.DataFrame(KNOWN_EVENTS, columns=["date", "event"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def nearby_event(date: pd.Timestamp, window_days: int = 3) -> str:
    cal = event_calendar()
    if cal.empty:
        return ""
    delta = (cal.index - date).days
    mask = abs(delta) <= window_days
    if not mask.any():
        return ""
    return "; ".join(cal.loc[mask, "event"].tolist())


def tag_alerts(alerts: pd.DataFrame, date_col: str = "date", window_days: int = 3) -> pd.DataFrame:
    if alerts.empty:
        alerts["nearby_event"] = []
        return alerts
    alerts = alerts.copy()
    alerts["nearby_event"] = [nearby_event(pd.Timestamp(d), window_days) for d in alerts[date_col]]
    return alerts
