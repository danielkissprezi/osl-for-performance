#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

measurements = np.loadtxt("frame-times.txt")
measurements = pd.Series(measurements)


def run(measurements):
    # for each index (i) total_time[i] = total time spent so far (sum(measurements[0..i]))
    total_time = np.cumsum(measurements)
    X = np.arange(len(measurements), dtype=float).reshape(-1, 1)

    regr = linear_model.LinearRegression()

    regr.fit(X, total_time)

    pred_total_time = regr.predict(X)

    mean = np.average(measurements)
    ols_estimate = regr.coef_[0]
    return {
        "mean": mean,
        "std": np.std(measurements),
        "ols_estimate": ols_estimate,
        "r2": r2_score(total_time, pred_total_time),
        "total_time": total_time,
        "pred": pred_total_time,
    }


def bootstrap(n, iters):
    for _ in range(iters):
        yield np.random.choice(n, n, replace=True)


s = run(measurements)
bs = {k: [] for k in s.keys()}

for idx in bootstrap(len(measurements), 1000):
    d = run(measurements[idx])
    for k, v in d.items():
        bs[k].append(v)


df = pd.DataFrame(
    {
        "label": ["OLS regression estimate", "R2 score", "mean", "stdev"],
        "min": [
            min(bs["ols_estimate"]),
            min(bs["r2"]),
            min(bs["mean"]),
            min(bs["std"]),
        ],
        "esitmate": [
            s["ols_estimate"],
            s["r2"],
            s["mean"],
            s["std"],
        ],
        "max": [
            max(bs["ols_estimate"]),
            max(bs["r2"]),
            max(bs["mean"]),
            max(bs["std"]),
        ],
    }
)

with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df)

ax = measurements.plot.kde(label="kde")
xlim = plt.xlim()
plt.xlim((max(0.0, xlim[0]), xlim[1]))  # perf data is always positive
(ymin, ymax) = ax.get_ylim()
plt.vlines(s["ols_estimate"], ymin, ymax, color="red", label="ols estimate")
plt.vlines(s["mean"], ymin, ymax, color="blue", label="mean")
plt.legend()

plt.savefig("docs/kde.png")
plt.figure()

for n in [10, 20, 100, 200, 500]:
    measurements.plot.hist(bins=n, label=f"bins={n}")
    plt.legend()
    plt.savefig(f"docs/histogram-bin-{n}.png")
    plt.figure()

X = np.arange(len(measurements))
plt.scatter(X, measurements)

plt.savefig("docs/measures.png")
plt.figure()

plt.scatter(X, s["total_time"], color="black")
plt.plot(X, s["pred"], linewidth=3)
plt.savefig("docs/ols.png")
