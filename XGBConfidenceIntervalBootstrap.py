import numpy as np
from xgboost import XGBRegressor
from tqdm import tqdm


class XGBConfidenceIntervalBootstrap:
    def __init__(self, n_regressors=100, n_common_trees=0, sample_rate=1.0, **xgb_args):
        self.n_regressors = n_regressors
        self.n_common_trees = n_common_trees
        self.sample_rate = sample_rate
        self.xgb_args = xgb_args
        self.base_regressor = None
        self.regressors = []

    def fit(self, X, y):
        # gpu_args = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor', 'gpu_id': 0, 'n_jobs': 16}
        if self.n_common_trees:
            base_regressor_args = {'objective': 'reg:squarederror', 'n_estimators': self.n_common_trees}
            self.base_regressor = XGBRegressor(**base_regressor_args)
            self.base_regressor.fit(X, y, verbose=False)
            self.base_regressor.save_model('base.model')

        for i in tqdm(range(self.n_regressors)):
            regressor = XGBRegressor(**self.xgb_args)
            n_samples = int(len(X) * self.sample_rate)
            sample_indexes = np.random.choice(range(len(X)), n_samples, replace=True)
            train_args = {}
            if self.n_common_trees:
                train_args['xgb_model'] = 'base.model'
            regressor.fit(X[sample_indexes], y[sample_indexes], verbose=False, **train_args)
            self.regressors.append(regressor)

    def predict(self, X):
        result = np.array([r.predict(X) for r in self.regressors])
        mean = result.mean(axis=0)
        lower = np.quantile(result, 0.05, axis=0)
        upper = np.quantile(result, 0.95, axis=0)
        return mean, lower, upper
