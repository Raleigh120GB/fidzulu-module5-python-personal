import pandas as pd
from typing import Dict, Tuple


class TrainTestSplitter:
    """Simple time-series train/test splitter per product.

    Splits each product's rows by chronological order using the provided
    test_ratio (fraction of rows reserved for testing).
    """

    def __init__(self, df: pd.DataFrame, test_ratio: float = 0.2):
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df.copy()
        # defensive: coerce test_ratio to float and clamp to [0,1]
        try:
            tr = float(test_ratio)
        except Exception:
            tr = 0.2
        self.test_ratio = min(max(tr, 0.0), 1.0)

    def split(self) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
        train_sets = {}
        test_sets = {}

        for prod_id, group in self.df.groupby("prod_id"):
            # validate expected columns exist
            if "start_date" not in group.columns or "base_price" not in group.columns:
                train_sets[prod_id] = pd.DataFrame()
                test_sets[prod_id] = pd.DataFrame()
                continue
            grp = group.sort_values("start_date").reset_index(drop=True)
            n = len(grp)
            if n == 0:
                train_sets[prod_id] = grp.copy()
                test_sets[prod_id] = grp.copy()
                continue

            test_count = max(0, int(round(n * self.test_ratio)))
            split_idx = n - test_count
            if split_idx <= 0:
                train = grp.iloc[:0].reset_index(drop=True)
                test = grp.copy()
            else:
                train = grp.iloc[:split_idx].reset_index(drop=True)
                test = grp.iloc[split_idx:].reset_index(drop=True)

            # IQR outlier filter on train split only, only if quartiles sensible
            if len(train) > 0:
                try:
                    q1 = train["base_price"].quantile(0.25)
                    q3 = train["base_price"].quantile(0.75)
                    iqr = q3 - q1
                    if pd.notna(iqr) and iqr > 0:
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        train = train[
                            (train["base_price"] >= lower) & (train["base_price"] <= upper)
                        ].reset_index(drop=True)
                except Exception:
                    # if any problem computing IQR, skip filtering
                    pass

            train_sets[prod_id] = train
            test_sets[prod_id] = test

        return train_sets, test_sets
