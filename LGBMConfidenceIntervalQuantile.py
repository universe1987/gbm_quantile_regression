from lightgbm import LGBMRegressor


class LGBMConfidenceIntervalQuantile:
    def __init__(self, **lgbm_args):
        self.mean_regressor = LGBMRegressor(**lgbm_args)
        self.upper_regressor = LGBMRegressor(objective='quantile', alhpa=0.95, **lgbm_args)
        self.lower_regressor = LGBMRegressor(objective='quantile', alpha=0.05, **lgbm_args)

    def fit(self, X, y):
        self.mean_regressor.fit(X, y)
        self.lower_regressor.fit(X, y)
        self.upper_regressor.fit(X, y)

    def predict(self, X):
        mean = self.mean_regressor.predict(X)
        lower = self.lower_regressor.predict(X)
        upper = self.upper_regressor.predict(X)
        return mean, lower, upper
