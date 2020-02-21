import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams
from XGBConfidenceIntervalBootstrap import XGBConfidenceIntervalBootstrap
from XGBConfidenceIntervalQuantile import XGBConfidenceIntervalQuantile
from LGBMConfidenceIntervalQuantile import LGBMConfidenceIntervalQuantile

rcParams['figure.figsize'] = 15, 6
sns.set(color_codes=True)


def generate_dataset(size):
    X = 2 * np.random.rand(size)
    # y = X ** 2 + np.random.normal(0, 1, len(X))
    y = X * np.sin(2 * np.pi * X) * (1 + np.random.normal(0, 1, len(X)))
    return X.reshape(-1, 1), y


def plot_prediction(X, y, mean, lower, upper, filename):
    indexes = np.argsort(X.flatten())
    x_coord = X.flatten()[indexes]
    y_lower = lower[indexes]
    y_upper = upper[indexes]
    y_truth = y[indexes]
    y_pred = mean[indexes]
    get_gaps(y_lower, y_upper)
    plt.fill_between(x_coord, y_lower, y_upper, facecolor='green', interpolate=True, alpha=0.3)
    plt.plot(x_coord, x_coord * np.sin(2 * np.pi * x_coord), label='truth')
    plt.plot(x_coord, y_truth, label='sample')
    plt.plot(x_coord, y_pred, label='pred')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def compare(X, y, mean1, lower1, upper1, mean2, lower2, upper2, output_name):
    indexes = np.argsort(X.flatten())
    x_coord = X.flatten()[indexes]
    y_truth = y[indexes]
    y_lower1 = lower1[indexes]
    y_upper1 = upper1[indexes]
    gap1 = y_upper1 - y_lower1
    y_pred1 = mean1[indexes]
    y_lower2 = lower2[indexes]
    y_upper2 = upper2[indexes]
    gap2 = y_upper2 - y_lower2
    y_pred2 = mean2[indexes]
    plt.plot(x_coord, x_coord * np.sin(2 * np.pi * x_coord) / 4, label='truth')
    plt.plot(x_coord, gap1, label='gap1')
    plt.plot(x_coord, gap2, label='gap2')
    plt.legend()
    plt.savefig(output_name)
    plt.close()


def get_gaps(lower, upper):
    gap1 = (upper - lower).mean()
    gap2 = np.quantile(upper - lower, 0.5)
    print(gap1, gap2)


def effect_of_sample_rate():
    xgb_args = {'objective': 'reg:squarederror', 'n_estimators': 100}
    train_x, train_y = generate_dataset(100000)
    test_x, test_y = generate_dataset(10000)
    result = []
    for sample_rate in np.linspace(0.2, 1, 17):
        bootstrap_regressor = XGBConfidenceIntervalBootstrap(sample_rate=sample_rate, **xgb_args)
        bootstrap_regressor.fit(train_x, train_y)
        mean, lower, upper = bootstrap_regressor.predict(test_x)
        gap = upper - lower
        gap_mean = gap.mean()
        gap_median = np.quantile(gap, 0.5)
        gap_std = gap.std()
        result.append([sample_rate, gap_mean, gap_median, gap_std])
    result = np.array(result)
    plt.plot(result[:, 0], result[:, 1], label='mean')
    plt.plot(result[:, 0], result[:, 2], label='median')
    plt.plot(result[:, 0], result[:, 3], label='std')
    plt.legend()
    plt.savefig('sample_rate.png')
    plt.close()
    result.tofile('result.npy')


def predict_with_confidence_interval():
    xgb_args = {'objective': 'reg:squarederror', 'n_estimators': 100}
    xgb_args_b = {'objective': 'reg:squarederror', 'n_estimators': 90}
    quantile_regressor = XGBConfidenceIntervalQuantile(**xgb_args)
    bootstrap_regressor = XGBConfidenceIntervalBootstrap(**xgb_args)
    bootstrap_regressor_b1 = XGBConfidenceIntervalBootstrap(n_common_trees=1, **xgb_args_b)
    bootstrap_regressor_b10 = XGBConfidenceIntervalBootstrap(n_common_trees=10, **xgb_args_b)
    bootstrap_regressor2 = XGBConfidenceIntervalBootstrap(sample_rate=0.5, **xgb_args)
    lgbm_quantile_regressor = LGBMConfidenceIntervalQuantile()
    train_x, train_y = generate_dataset(100000)
    quantile_regressor.fit(train_x, train_y)
    bootstrap_regressor.fit(train_x, train_y)
    bootstrap_regressor_b1.fit(train_x, train_y)
    bootstrap_regressor_b10.fit(train_x, train_y)
    bootstrap_regressor2.fit(train_x, train_y)
    lgbm_quantile_regressor.fit(train_x, train_y)
    test_x, test_y = generate_dataset(10000)
    quantile_pred = quantile_regressor.predict(test_x)
    plot_prediction(test_x, test_y, *quantile_pred, 'quantile.png')
    bootstrap_pred = bootstrap_regressor.predict(test_x)
    plot_prediction(test_x, test_y, *bootstrap_pred, 'bootstrap.png')
    bootstrap_pred_b1 = bootstrap_regressor_b1.predict(test_x)
    plot_prediction(test_x, test_y, *bootstrap_pred_b1, 'bootstrap_b1.png')
    bootstrap_pred_b10 = bootstrap_regressor_b10.predict(test_x)
    plot_prediction(test_x, test_y, *bootstrap_pred_b10, 'bootstrap_b1.png')
    bootstrap_pred2 = bootstrap_regressor2.predict(test_x)
    plot_prediction(test_x, test_y, *bootstrap_pred2, 'bootstrap2.png')
    lgbm_pred = lgbm_quantile_regressor.predict(test_x)
    plot_prediction(test_x, test_y, *lgbm_pred, 'lgbm.png')
    compare(test_x, test_y, *bootstrap_pred, *bootstrap_pred_b1, output_name='compare1.png')
    compare(test_x, test_y, *bootstrap_pred, *bootstrap_pred_b10, output_name='compare10.png')


if __name__ == '__main__':
    # effect_of_sample_rate()
    predict_with_confidence_interval()
