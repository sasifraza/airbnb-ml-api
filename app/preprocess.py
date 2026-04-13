import pandas as pd


FEATURE_COLUMNS = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

TARGET_COLUMN = "price"


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df[required_columns].copy()

    df = df.dropna()

    if df[TARGET_COLUMN].dtype == "object":
        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    df = df.dropna()

    df = df[df[TARGET_COLUMN] > 0]

    return df


def split_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y