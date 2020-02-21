from xgboost import XGBRegressor
from XGBQuantileRegressor import XGBQuantileRegressor


class XGBConfidenceIntervalQuantile:
    def __init__(self, **xgb_args):
        self.mean_regressor = XGBRegressor(**xgb_args)
        self.upper_regressor = XGBQuantileRegressor(quant_alpha=0.95, quant_delta=1.0, quant_thres=6.0, quant_var=3.2,
                                                    **xgb_args)
        self.lower_regressor = XGBQuantileRegressor(quant_alpha=0.05, quant_delta=1.0, quant_thres=5.0, quant_var=4.2,
                                                    **xgb_args)

    def fit(self, X, y):
        self.mean_regressor.fit(X, y)
        self.lower_regressor.fit(X, y)
        self.upper_regressor.fit(X, y)

    def predict(self, X):
        mean = self.mean_regressor.predict(X)
        lower = self.lower_regressor.predict(X)
        upper = self.upper_regressor.predict(X)
        return mean, lower, upper
