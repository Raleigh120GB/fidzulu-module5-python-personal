from src.fidzulu.business.train_test_splitter import TrainTestSplitter
import pandas as pd

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
    "base_price": ["10", "11.0", "12", "1000", "13", "14", "20", "21", "22", "23", "5000"],
})

splitter = TrainTestSplitter(sample, test_ratio=0.2)
concatenated, train_splits, test_splits = splitter.split_with_median_iqr_concat()
print(concatenated[['prod_id','start_date','base_price','t']].sort_values(['prod_id','start_date']).to_string(index=False))
