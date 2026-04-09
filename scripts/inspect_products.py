import sys, os
import pandas as pd
repo = os.path.abspath(os.path.join(os.getcwd()))
if repo not in sys.path:
    sys.path.insert(0, repo)
from src.fidzulu.business.price_wrangler import PriceDataWrangler
from src.fidzulu.business.train_test_splitter import TrainTestSplitter
cache='notebooks/vegetable_prices_cache.json'
raw=None
if os.path.exists(cache):
    try:
        raw = pd.read_json(cache, orient='records')
    except Exception as e:
        print('Failed reading cache:', e)
        raw=None
if raw is None:
    print('No cached raw_data available to inspect')
    raise SystemExit(0)
wr=PriceDataWrangler(raw)
df, _ = wr.wrangle()
splitter=TrainTestSplitter(df, test_ratio=0.2)
conc, trains, tests = splitter.split_with_median_iqr_concat()
for pid in (105,106):
    print('\n---- PROD', pid, '----')
    tr = trains.get(pid)
    te = tests.get(pid)
    print('TRAIN ROWS:', None if tr is None else len(tr))
    if tr is not None and not tr.empty:
        print(tr[['start_date','base_price','t']].to_string(index=False))
    print('TEST ROWS:', None if te is None else len(te))
    if te is not None and not te.empty:
        print(te[['start_date','base_price','t']].to_string(index=False))
