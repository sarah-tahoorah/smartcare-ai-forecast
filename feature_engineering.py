import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, net pressure, and calendar features."""
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["net_pressure"] = out["transfers"] - out["discharges"]

    # Lag features
    for lag in [1, 7, 14]:
        out[f"care_lag_{lag}"] = out["children_in_care"].shift(lag)
        out[f"pressure_lag_{lag}"] = out["net_pressure"].shift(lag)

    # Rolling features
    out["care_roll_7"] = out["children_in_care"].rolling(7).mean()
    out["care_roll_14"] = out["children_in_care"].rolling(14).mean()
    out["care_roll_std_14"] = out["children_in_care"].rolling(14).std()

    # Calendar features
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month

    return out
