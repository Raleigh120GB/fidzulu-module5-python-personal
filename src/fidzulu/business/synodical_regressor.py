import numpy as np


class SynodicalRegressor:
    """Fourier-basis linear regressor with 5 named coefficients.

    Features: t (linear), sin/cos annual cycle, sin/cos quarterly cycle.
    All coefficients are accessible as named attributes so that
    get_model_parameters() can expose them individually.
    """

    def __init__(self):
        self.coef_t = None
        self.coef_sin_year = None
        self.coef_cos_year = None
        self.coef_sin_q = None
        self.coef_cos_q = None
        self.intercept_ = None
        # coef_ list kept for compatibility: [coef_t, sin_year, cos_year, sin_q, cos_q]
        self.coef_ = None

    def _build_features(self, t_array):
        t = np.asarray(t_array, dtype=float)
        sin_year = np.sin(2 * np.pi * t / 365.25)
        cos_year = np.cos(2 * np.pi * t / 365.25)
        sin_q = np.sin(4 * np.pi * t / 365.25)
        cos_q = np.cos(4 * np.pi * t / 365.25)
        return np.column_stack([t, sin_year, cos_year, sin_q, cos_q, np.ones_like(t)])

    def fit(self, X, y):
        # Defensive checks: X and y must be array-like and same length
        try:
            x_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
        except Exception:
            raise TypeError("X and y must be numeric array-like inputs")

        if x_arr.ndim != 1:
            x_arr = x_arr.ravel()

        if len(x_arr) == 0 or len(y_arr) == 0 or len(x_arr) != len(y_arr):
            # insufficient data: initialize to safe defaults
            self.coef_t = 0.0
            self.coef_sin_year = 0.0
            self.coef_cos_year = 0.0
            self.coef_sin_q = 0.0
            self.coef_cos_q = 0.0
            self.intercept_ = float(np.mean(y_arr) if len(y_arr) > 0 else 0.0)
            self.coef_ = [self.coef_t, self.coef_sin_year, self.coef_cos_year,
                          self.coef_sin_q, self.coef_cos_q]
            return

        X_feat = self._build_features(x_arr)
        try:
            beta, _, _, _ = np.linalg.lstsq(X_feat, y_arr, rcond=None)
        except Exception:
            beta = np.zeros(6)

        self.coef_t = float(beta[0])
        self.coef_sin_year = float(beta[1])
        self.coef_cos_year = float(beta[2])
        self.coef_sin_q = float(beta[3])
        self.coef_cos_q = float(beta[4])
        self.intercept_ = float(beta[5])
        self.coef_ = [self.coef_t, self.coef_sin_year, self.coef_cos_year,
                      self.coef_sin_q, self.coef_cos_q]

    def predict(self, X):
        if self.coef_t is None:
            raise RuntimeError("Model is not fitted")
        X_feat = self._build_features(X)
        beta = np.array([self.coef_t, self.coef_sin_year, self.coef_cos_year,
                         self.coef_sin_q, self.coef_cos_q, self.intercept_])
        return X_feat @ beta
