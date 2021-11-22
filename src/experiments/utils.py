import ctypes
import io
import json
import os
import sys
import tempfile
from contextlib import contextmanager

import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def log_json(a, label):
    mlflow.log_text(json.dumps(a), f"{label}.json")


def log_array(a, label):
    mlflow.log_text(pd.DataFrame(a).to_csv(), f"{label}.csv")


# TODO Reduce duplication with evaluation.*
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

    if var is not None:
        var_ = var.ravel()[perm]
        std = np.sqrt(var_)
        ax.plot(X_test_, y_test_ - std, c="b", linestyle="dotted")
        ax.plot(X_test_, y_test_ + std, c="b", linestyle="dotted")
        ax.fill_between(X_test_, y_test_ - std, y_test_ + std, alpha=0.3)

    return fig, ax


def save_plot(fname, fig):
    # store the figure (e.g. so we can run headless)
    fig_folder = "plots"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    fig_file = f"{fig_folder}/{fname}-{mlflow.active_run().info.run_id}.pdf"
    print(f"Storing plot in {fig_file}")
    fig.savefig(fig_file)
    mlflow.log_artifact(fig_file)


libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')


@contextmanager
def stdout_redirector(stream):
    """
    Copied from https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/ .
    """
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)
