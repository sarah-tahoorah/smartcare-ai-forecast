import numpy as np
import pandas as pd

def load_or_generate_data(start_date: str = "2023-01-01", end_date: str = "2026-03-01") -> pd.DataFrame:
    """Generate a realistic synthetic daily dataset for care load and flows."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)

    # Seasonality components
    weekly = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)
    yearly = 60 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    trend = np.linspace(0, 120, n)

    # Transfers and discharges with noise
    transfers = 180 + 0.15 * trend + 0.3 * yearly + rng.normal(0, 15, n)
    discharges = 170 + 0.12 * trend + 0.25 * yearly + rng.normal(0, 14, n)

    # Care load evolves with net pressure
    net_pressure = transfers - discharges
    base = 12500 + 0.8 * trend + weekly + yearly
    care_load = base + np.cumsum(net_pressure * 0.08) + rng.normal(0, 25, n)

    df = pd.DataFrame(
        {
            "date": dates,
            "children_in_care": np.round(care_load, 0),
            "transfers": np.round(transfers, 0),
            "discharges": np.round(discharges, 0),
        }
    )
    df["children_in_care"] = df["children_in_care"].clip(lower=9000)
    df["transfers"] = df["transfers"].clip(lower=50)
    df["discharges"] = df["discharges"].clip(lower=40)

    return df
