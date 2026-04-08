import numpy as np
import pandas as pd
from typing import Dict

from src.fidzulu.business.price_wrangler import PriceDataWrangler
from src.fidzulu.business.train_test_splitter import TrainTestSplitter
from src.fidzulu.business.synodical_regressor import SynodicalRegressor


class PriceForecastService:
    def __init__(self, raw_data: Dict, test_ratio: float = 0.2):
        self.raw_data = raw_data
        self.test_ratio = float(test_ratio)
        self.wrangler = PriceDataWrangler(raw_data)
        self.df, self.feedback = self.wrangler.wrangle()

        splitter = TrainTestSplitter(self.df, test_ratio=self.test_ratio)
        self.train_sets, self.test_sets = splitter.split()

        self.models: Dict = {}
        self._base_dates: Dict = {}
        self._train_models()

    def _train_models(self):
        for pid, train_df in self.train_sets.items():
            model = SynodicalRegressor()
            if len(train_df) == 0:
                self.models[pid] = model
                self._base_dates[pid] = pd.Timestamp.today()
                continue

            base_date = train_df["start_date"].min()
            self._base_dates[pid] = base_date
            x = (train_df["start_date"] - base_date).dt.days.values
            y = train_df["base_price"].values
            model.fit(x, y)
            self.models[pid] = model

    def combined_train_df(self) -> pd.DataFrame:
        if not self.train_sets:
            return pd.DataFrame()
        frames = []
        for pid, df in self.train_sets.items():
            f = df.copy()
            f["prod_id"] = pid
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def evaluate(self, prod_id: int) -> Dict[str, float]:
        if prod_id not in self.models:
            raise ValueError(f"No model found for product {prod_id}")
        model = self.models[prod_id]
        test_df = self.test_sets.get(prod_id)
        if test_df is None or len(test_df) == 0:
            return {"mae": 0.0, "rmse": 0.0}

        base_date = self._base_dates.get(prod_id, test_df["start_date"].min())
        x = (test_df["start_date"] - base_date).dt.days.values
        y_true = test_df["base_price"].values
        y_pred = model.predict(x)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return {"mae": mae, "rmse": rmse}

    def get_evaluation(self, prod_id: int) -> Dict[str, float]:
        if prod_id not in self.models:
            raise ValueError(f"No model found for product {prod_id}")
        model = self.models[prod_id]
        test_df = self.test_sets.get(prod_id)
        if test_df is None or len(test_df) == 0:
            return {"mae": 0.0, "mse": 0.0, "rmse": 0.0}

        base_date = self._base_dates.get(prod_id, test_df["start_date"].min())
        x = (test_df["start_date"] - base_date).dt.days.values
        y_true = test_df["base_price"].values
        y_pred = model.predict(x)

        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        return {"mae": mae, "mse": mse, "rmse": rmse}

    def get_model_parameters(self, prod_id: int) -> Dict:
        if prod_id not in self.models:
            raise ValueError(f"No model found for product {prod_id}")
        model = self.models[prod_id]

        def _val(v):
            return float(v) if v is not None else 0.0

        return {
            "intercept": _val(model.intercept_),
            "coef_t": _val(model.coef_t),
            "coef_sin_year": _val(model.coef_sin_year),
            "coef_cos_year": _val(model.coef_cos_year),
            "coef_sin_q": _val(model.coef_sin_q),
            "coef_cos_q": _val(model.coef_cos_q),
            "base_min_date": self._base_dates.get(prod_id, pd.Timestamp.today()),
        }

    def forecast(self, prod_id: int, horizon_days: int = 180) -> pd.DataFrame:
        if prod_id not in self.models:
            raise ValueError(f"No model found for product {prod_id}")
        if horizon_days <= 0:
            return pd.DataFrame(columns=["date", "pred_price"])

        model = self.models[prod_id]
        train_df = self.train_sets.get(prod_id)
        if train_df is None or len(train_df) == 0:
            last_date = pd.Timestamp.today()
        else:
            last_date = train_df["start_date"].max()

        start = pd.to_datetime(last_date) + pd.Timedelta(days=1)
        dates = pd.date_range(start=start, periods=horizon_days, freq="D")

        base_date = self._base_dates.get(prod_id, pd.to_datetime(last_date))
        x = (dates - base_date).days.values
        preds = model.predict(x)

        return pd.DataFrame({"date": dates, "pred_price": preds})

