import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple

# --- Existing Wrangling Class ---
class PriceDataWrangler:
    def __init__(self, raw_data: Dict):
        if raw_data is None:
            raise ValueError("raw_data must be a non-empty dict")
        if not isinstance(raw_data, dict):
            raise TypeError("raw_data must be a dict mapping product ids to data")
        self.raw_data = raw_data
        self.category_id = raw_data.get("CategoryID")
        if self.category_id is None:
            raise ValueError("raw_data missing required key 'CategoryID'")
        self.df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        records = []
        for prod_id, data in self.raw_data.items():
            if prod_id == "CategoryID":
                continue
            if not isinstance(data, dict):
                continue
            prices = data.get("prices")
            starts = data.get("start_dates")
            ends = data.get("end_dates")
            if prices is None or starts is None or ends is None:
                # skip malformed product entries
                continue
            # coerce to lists
            try:
                prices_list = list(prices)
                starts_list = list(starts)
                ends_list = list(ends)
            except Exception:
                continue

            # lengths must match; if not, use shortest
            length = min(len(prices_list), len(starts_list), len(ends_list))
            if length == 0:
                continue

            for i in range(length):
                price = prices_list[i]
                start = starts_list[i]
                end = ends_list[i]
                # coerce price to float when possible
                try:
                    base_price = float(price)
                except Exception:
                    # skip non-numeric prices
                    continue
                # parse dates; invalid dates become NaT
                start_ts = pd.to_datetime(start, errors="coerce")
                end_ts = pd.to_datetime(end, errors="coerce")
                records.append({
                    "prod_id": prod_id,
                    "base_price": base_price,
                    "start_date": start_ts,
                    "end_date": end_ts
                })
        return pd.DataFrame(records)

    def wrangle(self) -> Tuple[pd.DataFrame, Dict]:
        feedback = {}
        df = self.df.copy()
        
        # Guard: if no product rows at all
        if df.empty or "base_price" not in df.columns:
            raise ValueError("No valid product data found in raw input.")

        # Remove rows with non-positive prices
        invalid_prices = df[df["base_price"] <= 0]
        if not invalid_prices.empty:
            feedback["invalid_prices"] = f"Removed {len(invalid_prices)} rows with non-positive prices."
            df = df[df["base_price"] > 0]

        # Remove rows with missing or invalid dates
        missing_dates = df[df["start_date"].isna() | df["end_date"].isna()]
        if not missing_dates.empty:
            feedback["missing_dates"] = f"Removed {len(missing_dates)} rows with missing start/end dates."
            df = df.dropna(subset=["start_date", "end_date"])

        # Guard: if all rows dropped
        if df.empty:
            raise ValueError("No valid rows remain after dropping missing dates.")

       
        # Remove rows where end_date is before start_date
        bad_ranges = df[df["end_date"] < df["start_date"]]
        if not bad_ranges.empty:
            feedback["bad_ranges"] = f"Removed {len(bad_ranges)} rows where end_date precedes start_date."
            df = df[df["end_date"] >= df["start_date"]]

        # Guard: if all rows dropped
        if df.empty:
            raise ValueError("No valid rows remain after removing bad date ranges.")    

        return df.reset_index(drop=True), feedback

