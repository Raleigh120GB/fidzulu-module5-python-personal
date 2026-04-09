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
        # Map common alternative column names to the expected `base_price`
        if "base_price" not in self.df.columns:
            alt_price_cols = ["price", "avg_price", "value", "unit_price"]
            for alt in alt_price_cols:
                if alt in self.df.columns:
                    self.df["base_price"] = self.df[alt]
                    print(f"[TrainTestSplitter] mapped alternative price column '{alt}' to 'base_price'")
                    break
        # Ensure `base_price` is numeric (convert from text if needed)
        if "base_price" in self.df.columns:
            before_nonnull = self.df["base_price"].notnull().sum()
            # coerce strings to numeric; invalid parsing -> NaN
            self.df["base_price"] = pd.to_numeric(self.df["base_price"], errors="coerce")
            after_nonnull = self.df["base_price"].notnull().sum()
            if before_nonnull != after_nonnull:
                print(f"[TrainTestSplitter] coerced base_price to numeric: non-null before={before_nonnull}, after={after_nonnull}")
        # Map common alternative column names to the expected `start_date`
        if "start_date" not in self.df.columns:
            alt_date_cols = ["date", "obs_date", "timestamp"]
            for alt in alt_date_cols:
                if alt in self.df.columns:
                    self.df["start_date"] = self.df[alt]
                    print(f"[TrainTestSplitter] mapped alternative date column '{alt}' to 'start_date'")
                    break
        # Ensure `start_date` is a datetime for reliable sorting
        if "start_date" in self.df.columns:
            self.df["start_date"] = pd.to_datetime(self.df["start_date"], errors="coerce")
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
                # create numeric time column 't' as days since the first date for this product
            try:
                base = grp["start_date"].min()
                if pd.notna(base):
                    grp["t"] = (grp["start_date"] - base).dt.days.astype("Int64")
                else:
                    grp["t"] = pd.NA
            except Exception:
                grp["t"] = pd.NA
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

    def split_with_median_iqr_concat(self) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
        """Split per product, remove outliers using IQR around the median,
        and return a concatenated train DataFrame plus per-product train/test dicts.

        Outliers are defined as values outside `median +/- 1.5 * IQR` where
        `IQR = Q3 - Q1` computed on the train split for each product.
        Prints summary information for verification.
        """
        train_sets: Dict[int, pd.DataFrame] = {}
        test_sets: Dict[int, pd.DataFrame] = {}
        processed = 0

        for prod_id, group in self.df.groupby("prod_id"):
            # validate expected columns exist
            if "start_date" not in group.columns or "base_price" not in group.columns:
                print(f"[TrainTestSplitter] product {prod_id}: missing columns, skipping")
                train_sets[prod_id] = pd.DataFrame()
                test_sets[prod_id] = pd.DataFrame()
                continue

            grp = group.sort_values("start_date").reset_index(drop=True)
                # create numeric time column 't' as days since the first date for this product
            try:
                base = grp["start_date"].min()
                if pd.notna(base):
                    grp["t"] = (grp["start_date"] - base).dt.days.astype("Int64")
                else:
                    grp["t"] = pd.NA
            except Exception:
                grp["t"] = pd.NA
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

            before = len(train)
            # Apply median-based IQR filtering to the train split only
            if len(train) > 0:
                try:
                    q1 = train["base_price"].quantile(0.25)
                    q3 = train["base_price"].quantile(0.75)
                    iqr = q3 - q1
                    if pd.notna(iqr) and iqr > 0:
                        med = train["base_price"].median()
                        lower = med - 1.5 * iqr
                        upper = med + 1.5 * iqr
                        train = train[
                            (train["base_price"] >= lower) & (train["base_price"] <= upper)
                        ].reset_index(drop=True)
                except Exception as exc:
                    print(f"[TrainTestSplitter] product {prod_id}: IQR filtering failed: {exc}")

            after = len(train)
            print(f"[TrainTestSplitter] product {prod_id}: train rows before={before}, after={after}, test rows={len(test)}")

            train_sets[prod_id] = train
            test_sets[prod_id] = test
            processed += 1

        # Concatenate all non-empty train splits
        non_empty_trains = [df for df in train_sets.values() if df is not None and not df.empty]
        if non_empty_trains:
            concatenated = pd.concat(non_empty_trains, ignore_index=True)
        else:
            concatenated = pd.DataFrame(columns=self.df.columns)

        print(f"[TrainTestSplitter] processed {processed} products, concatenated train shape={concatenated.shape}")
        return concatenated, train_sets, test_sets


if __name__ == "__main__":
    # Small self-test when module is run as a script
    print("TrainTestSplitter self-test starting...")
    sample = pd.DataFrame({
        "prod_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "start_date": pd.to_datetime([
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
            "2020-06-01",
            "2020-01-15",
            "2020-02-15",
            "2020-03-15",
            "2020-04-15",
            "2020-05-15",
        ]),
        # include some prices as strings to validate coercion
        "base_price": ["10", "11.0", "12", "1000", "13", "14", "20", "21", "22", "23", "5000"],
    })

    splitter = TrainTestSplitter(sample, test_ratio=0.2)
    concatenated, train_splits, test_splits = splitter.split_with_median_iqr_concat()

    print("Concatenated train shape:", concatenated.shape)
    print("Per-product train sizes:", {k: len(v) for k, v in train_splits.items()})
    print("Per-product test sizes:", {k: len(v) for k, v in test_splits.items()})
    print("TrainTestSplitter self-test completed.")
