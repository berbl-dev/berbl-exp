import json
import os
import tempfile

import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def log_json(a, label):
    mlflow.log_text(json.dumps(a), f"{label}.json")


def log_array(a, label):
    f = tempfile.NamedTemporaryFile(prefix=f"{label}-", suffix=f".csv")
    pd.DataFrame(a).to_csv(f.name)
    mlflow.log_artifact(f.name)
    f.close()


def plot_prediction(X,
                    y,
                    X_test,
                    y_test,
                    var=None,
                    X_denoised=None,
                    y_denoised=None):
    fig, ax = plt.subplots()

    # plot input data
    ax.plot(X.ravel(), y.ravel(), "r+")

    if X_denoised is not None and y_denoised is not None:
        # plot denoised input data for visual reference
        ax.plot(X_denoised.ravel(), y_denoised.ravel(), "k--")

    # plot test data
    X_test_ = X_test.ravel()
    perm = np.argsort(X_test_)
    X_test_ = X_test_[perm]
    y_test_ = y_test.ravel()[perm]
    ax.plot(X_test_, y_test_, "b-")

    if var:
        var_ = var.ravel()[perm]
        std = np.sqrt(var_)
        ax.plot(X_test_, y_test_ - std, "b--", linewidth=0.5)
        ax.plot(X_test_, y_test_ + std, "b--", linewidth=0.5)
        ax.fill_between(X_test_, y_test_ - std, y_test_ + std, alpha=0.2)

    return fig, ax


def save_plot(fig, seed):
    # store the figure (e.g. so we can run headless)
    fig_folder = "latest-final-approximations"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig_file = f"{fig_folder}/Final approximation {seed}.pdf"
    print(f"Storing final approximation figure in {fig_file}")
    fig.savefig(fig_file)
    mlflow.log_artifact(fig_file)
