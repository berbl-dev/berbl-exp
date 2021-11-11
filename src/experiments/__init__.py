import abc
import importlib

import matplotlib.pyplot  # type: ignore
import mlflow  # type: ignore
import numpy  # type: ignore
from experiments.utils import log_array
from sklearn.compose import TransformedTargetRegressor  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def get_data(module, data_seed):
    task_path = f"tasks.{module}"
    task_mod = importlib.import_module(task_path)
    return task_mod.data(data_seed)


def experiment_name(algorithm, module):
    return f"{algorithm}.{module}"


def maybe_override(params, param, value):
    if value is not None and params[param] != value:
        print(f"Warning: Overriding {param}={params[param]} with "
              f"{param}={value}")
        params[param] = value


class GenerativePipeline(Pipeline):
    def predict_mean_var(self, X, **predict_mean_var_params):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][1].predict_mean_var(Xt,
                                                  **predict_mean_var_params)

    def predict_distribution(self, X, **predict_distribution_params):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][1].predict_distribution(
            Xt, **predict_distribution_params)


class Experiment(abc.ABC):
    @property
    @abc.abstractmethod
    def algorithm(self):
        pass

    def __init__(self, module, seed, data_seed, standardize, show):
        self.experiment_name = experiment_name(self.algorithm, module)

        self.seed = seed

        exp_path = f"experiments.{self.experiment_name}"
        exp_mod = importlib.import_module(exp_path)
        self.params = exp_mod.params

        self.data = get_data(exp_mod.task, data_seed)

        self.standardize = standardize
        self.show = show

    def run(self, **kwargs):
        for key in kwargs:
            maybe_override(self.params, key, kwargs[key])

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run() as run:
            print(f"Started experiment: {self.experiment_name}")
            X = self.data["X"]
            y = self.data["y"]
            X_test = self.data["X_test"]
            y_test_true = self.data["y_test_true"]
            X_denoised = self.data["X_denoised"]
            y_denoised = self.data["y_denoised"]

            mlflow.log_param("seed", self.seed)
            mlflow.log_param("train.size", len(X))
            mlflow.log_param("standardize", self.standardize)
            mlflow.log_params(self.params)
            log_array(X, "X")
            log_array(y, "y")
            log_array(X_test, "X_test")
            log_array(y_test_true, "y_test_true")
            log_array(X_denoised, "X_denoised")
            log_array(y_denoised, "y_denoised")

            self.init_estimator()

            if self.standardize:
                self.estimator = GenerativePipeline([
                    ("standardscaler", StandardScaler()),
                    ("transfromedtargetregressor",
                     TransformedTargetRegressor(regressor=self.estimator,
                                                transformer=StandardScaler()))
                ])

            self.estimator.fit(X, y)

            self.evaluate(X, y, X_test, y_test_true, X_denoised, y_denoised)

    @abc.abstractmethod
    def init_estimator(self):
        # Use self.seed here!
        self.estimator = ...

    @abc.abstractmethod
    def evaluate(self, X, y, X_test, y_test_true, X_denoised, y_denoised):
        ...
