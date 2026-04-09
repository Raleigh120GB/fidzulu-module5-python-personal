"""Demo: fit double_sin_trend_model to synthetic seasonal price data.

Run with:

$env:PYTHONPATH='D:\Associate\Repo-Module5-FidZulu-PythonML-Practice'; python scripts/fit_double_sin_demo.py

"""
import sys, os
repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
from scipy.optimize import curve_fit
from src.fidzulu.business.synodical_regressor import double_sin_trend_model

# synthetic time (days)
t = np.arange(0, 365 * 2)  # two years of daily data

# true params
intercept_true = 10.0
slope_true = 0.001  # slow upward trend per day
A1_true, freq1_true, phase1_true = 2.0, 1/365.0, 0.1  # annual
A2_true, freq2_true, phase2_true = 0.5, 2/365.0, -0.5  # semiannual

rng = np.random.default_rng(42)
noise = rng.normal(scale=0.3, size=t.shape)

y_true = double_sin_trend_model(t,
                                intercept_true, slope_true,
                                A1_true, freq1_true, phase1_true,
                                A2_true, freq2_true, phase2_true)

y_obs = y_true + noise

# initial guesses: intercept ~ mean, slope 0, amplitudes ~ std, frequencies annual+semiannual
p0 = [np.mean(y_obs), 0.0,
      1.0, 1/365.0, 0.0,
      0.5, 2/365.0, 0.0]

print('Starting fit with p0 =', p0)
params, cov = curve_fit(double_sin_trend_model, t, y_obs, p0=p0, maxfev=20000)

param_names = ['intercept','slope','A1','freq1','phase1','A2','freq2','phase2']
print('\nFitted parameters:')
for name, val, true in zip(param_names, params, [intercept_true, slope_true, A1_true, freq1_true, phase1_true, A2_true, freq2_true, phase2_true]):
    print(f"{name:10s}: {val:.6g}    (true: {true})")

# quick residual RMS
resid = y_obs - double_sin_trend_model(t, *params)
print('\nResidual RMS:', np.sqrt(np.mean(resid**2)))
