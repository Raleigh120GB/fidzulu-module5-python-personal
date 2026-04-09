import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

from .synodical_regressor import double_sin_trend_model
from .synodical_regressor import single_sin_trend_model


def fit_double_sin_per_product(train_splits: Dict[int, pd.DataFrame], *, maxfev: int = 20000) -> Dict[int, Dict[str, Any]]:
    """Fit `double_sin_trend_model` to each product's train split.

    Args:
        train_splits: mapping prod_id -> DataFrame containing at least
            `t` (numeric days since first date) and `base_price` (numeric).
        maxfev: maximum function evaluations passed to `curve_fit`.

    Returns:
        dict mapping prod_id -> result dict with keys:
          - params: fitted parameter array (or None)
          - cov: covariance matrix (or None)
          - success: bool
          - message: error message when not successful
          - n: number of points used
    """
    results: Dict[int, Dict[str, Any]] = {}

    for prod_id, df in train_splits.items():
        res = {"params": None, "cov": None, "success": False, "message": None, "n": 0}
        if df is None or df.empty:
            res["message"] = "empty or missing DataFrame"
            results[prod_id] = res
            continue

        # Ensure `t` exists: compute from `start_date` if necessary
        if "t" not in df.columns:
            if "start_date" in df.columns:
                try:
                    base = pd.to_datetime(df["start_date"]).min()
                    df = df.copy()
                    df["t"] = (pd.to_datetime(df["start_date"]) - base).dt.days
                except Exception:
                    res["message"] = "unable to compute 't' from 'start_date'"
                    results[prod_id] = res
                    continue
            else:
                res["message"] = "missing 't' and no 'start_date' to compute it"
                results[prod_id] = res
                continue

        # Ensure `base_price` exists: try common alternative column names
        if "base_price" not in df.columns:
            alt_candidates = ["price", "avg_price", "value", "unit_price"]
            found = False
            for alt in alt_candidates:
                if alt in df.columns:
                    df = df.copy()
                    df["base_price"] = df[alt]
                    found = True
                    break
            if not found:
                res["message"] = "missing 'base_price' and no alternative found"
                results[prod_id] = res
                continue

        t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["base_price"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(t) & np.isfinite(y)

        if mask.sum() < 6:
            res["message"] = f"insufficient valid points ({mask.sum()})"
            results[prod_id] = res
            continue

        t_use = t[mask]
        y_use = y[mask]

        # sensible linear initial guesses (intercept/slope)
        intercept0 = float(np.nanmean(y_use))
        slope0 = 0.0
        if np.ptp(t_use) > 0:
            slope0 = float((y_use[-1] - y_use[0]) / (t_use[-1] - t_use[0]))

        # amplitude heuristic
        amp_std = float(max(1e-6, np.nanstd(y_use)))

        # Estimate dominant frequency candidates via FFT on an interpolated grid
        try:
            ngrid = max(64, int(np.round(np.ptp(t_use))))
            if ngrid < 64:
                ngrid = 64
            tt_lin = np.linspace(t_use.min(), t_use.max(), ngrid)
            yy_lin = np.interp(tt_lin, t_use, y_use)
            fft = np.fft.rfft(yy_lin - np.mean(yy_lin))
            freqs = np.fft.rfftfreq(len(tt_lin), d=(tt_lin[1] - tt_lin[0]))
            power = np.abs(fft)
            # ignore zero frequency
            valid = freqs > 0
            candidate_freqs = freqs[valid][np.argsort(power[valid])[::-1]]
            # keep frequencies within reasonable bounds (0.0005 - 1 cycles/day)
            candidate_freqs = [float(f) for f in candidate_freqs if 5e-4 <= f <= 1.0]
        except Exception:
            candidate_freqs = []

        # prepare a list of frequency seeds to try (include annual by default)
        seeds = [1.0 / 365.0]
        for f in candidate_freqs:
            if f not in seeds:
                seeds.append(f)
            if len(seeds) >= 5:
                break

        # bounds for double-sin: amplitudes >=0, frequencies positive up to 1 cycle/day
        lower = [-np.inf, -np.inf, 0.0, 1e-6, -2*np.pi, 0.0, 1e-6, -2*np.pi]
        upper = [np.inf, np.inf, np.inf, 1.0, 2*np.pi, np.inf, 1.0, 2*np.pi]

        fit_ok = False
        last_exc = None

        # Try several initializations using candidate frequencies
        for freq1_guess in seeds:
            freq2_guess = min(1.0, max(1e-6, 2.0 * freq1_guess))
            p0 = [intercept0, slope0, amp_std, freq1_guess, 0.0, max(1e-6, amp_std / 2.0), freq2_guess, 0.0]
            try:
                params, cov = curve_fit(double_sin_trend_model, t_use, y_use, p0=p0, bounds=(lower, upper), maxfev=maxfev)
                res.update({"params": params, "cov": cov, "success": True, "n": int(mask.sum()), "message": f"double-sin fit seed freq1={freq1_guess:.6f}"})
                fit_ok = True
                break
            except Exception as exc:
                last_exc = exc

        # If double-sin failed, try relaxed frequency bounds and retry once
        if not fit_ok:
            try:
                lower_relaxed = lower.copy()
                upper_relaxed = upper.copy()
                upper_relaxed[3] = 2.0  # allow up to 2 cycles/day
                upper_relaxed[6] = 2.0
                p0 = [intercept0, slope0, amp_std, 1.0 / 365.0, 0.0, max(1e-6, amp_std / 2.0), 2.0 / 365.0, 0.0]
                params, cov = curve_fit(double_sin_trend_model, t_use, y_use, p0=p0, bounds=(lower_relaxed, upper_relaxed), maxfev=maxfev*2)
                res.update({"params": params, "cov": cov, "success": True, "n": int(mask.sum()), "message": "double-sin fit relaxed-bounds"})
                fit_ok = True
            except Exception as exc:
                last_exc = exc

        # If still failing, try single-sinusoid model as a fallback
        if not fit_ok:
            try:
                p0s = [intercept0, slope0, amp_std, seeds[0] if len(seeds) > 0 else 1.0 / 365.0, 0.0]
                lower_s = [-np.inf, -np.inf, 0.0, 1e-6, -2*np.pi]
                upper_s = [np.inf, np.inf, np.inf, 1.0, 2*np.pi]
                params_s, cov_s = curve_fit(single_sin_trend_model, t_use, y_use, p0=p0s, bounds=(lower_s, upper_s), maxfev=maxfev)
                # expand single-sin params to double-sin shaped output for compatibility
                params = [params_s[0], params_s[1], params_s[2], params_s[3], params_s[4], 0.0, params_s[3]*2.0, 0.0]
                res.update({"params": np.array(params), "cov": cov_s, "success": True, "n": int(mask.sum()), "message": "single-sin fallback"})
                fit_ok = True
            except Exception as exc:
                last_exc = exc

        # As a final attempt, try a robust least-squares fit (Huber loss) using least_squares
        if not fit_ok:
            try:
                # residual function for double-sin
                def _resid_double(p, tarr, yarr):
                    return double_sin_trend_model(tarr, *p) - yarr

                p0 = [intercept0, slope0, amp_std, seeds[0] if seeds else 1.0/365.0, 0.0, max(1e-6, amp_std/2.0), (seeds[0]*2.0) if seeds else 2.0/365.0, 0.0]
                lb = np.array(lower)
                ub = np.array(upper)
                # least_squares expects bounds as (lb, ub)
                ls = least_squares(_resid_double, x0=p0, args=(t_use, y_use), bounds=(lb, ub), max_nfev=maxfev, loss='huber')
                if ls.success:
                    params = ls.x
                    res.update({"params": params, "cov": None, "success": True, "n": int(mask.sum()), "message": "robust least_squares (huber)"})
                    fit_ok = True
                else:
                    last_exc = Exception('least_squares failed: ' + str(ls.message))
            except Exception as exc:
                last_exc = exc

        if not fit_ok:
            res["message"] = str(last_exc) if last_exc is not None else "fit failed"

        results[prod_id] = res

    return results
