import numpy as np
import pandas as pd


def adstock_transform(series, decay=0.5):
    """Geometric adstock — models carry-over effect of advertising."""
    adstocked = np.zeros(len(series))
    adstocked[0] = series.iloc[0]
    for i in range(1, len(series)):
        adstocked[i] = series.iloc[i] + decay * adstocked[i - 1]
    return adstocked


def saturation_transform(series, alpha=0.0001):
    """Exponential saturation — models diminishing returns."""
    return 1 - np.exp(-alpha * np.array(series))


def build_features(df, spend_cols, params):
    """Build adstock + saturation feature matrix from raw spend."""
    features = pd.DataFrame()
    for col in spend_cols:
        decay = params[f"{col}_decay"]
        alpha = params[f"{col}_alpha"]
        adstocked = adstock_transform(df[col], decay)
        features[col] = saturation_transform(adstocked, alpha)
    features["discount"] = df["discount"].values
    features["seasonality"] = df["seasonality"].values
    return features
