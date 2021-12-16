import abc
import importlib

import mlflow  # type: ignore
import sklearn.compose as compose  # type: ignore
import sklearn.pipeline as pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore

from .utils import log_array

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


class IdentityTargetRegressor(compose.TransformedTargetRegressor):
    """
    A ``TransformedTargetRegressor`` that uses ``func=None`` (i.e. identity).

    Adds (and passes through the pipeline to its final step)
    ``predict_mean_var``, ``predicts``, ``predict_distribution``.
    """
    def __init__(self, regressor=None):
        super().__init__(regressor=regressor, func=None)

    def predict_mean_var(self, X, **predict_mean_var_params):
        check_is_fitted(self)
        return self.regressor_.predict_mean_var(X)

    def predicts(self, X, **predicts_params):
        check_is_fitted(self)
        # TODO Note that we don't have the same resize/squeeze stuff in place
        # here as in
        # https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/compose/_target.py#L275
        return self.regressor_.predicts(X, **predicts_params)

    def predict_distribution(self, X, **predict_distribution_params):
        return self.regressor_.predict_distribution(X, **predict_distribution_params)


class StandardScaledTargetRegressor(compose.TransformedTargetRegressor):
    """
    A ``TransformedTargetRegressor`` that uses ``transformer=StandardScaler``.

    Adds (and passes through the pipeline to its final step)
    ``predict_mean_var``, ``predicts``, ``predict_distribution``. Note that
    ``predict_mean_var`` and ``predict_distribution`` are different for
    different transformers which is why ``StandardScaler`` is fixed here.
    """
    def __init__(self, regressor=None):
        super().__init__(regressor=regressor, transformer=StandardScaler())

    def predict_mean_var(self, X, **predict_mean_var_params):
        check_is_fitted(self)
        y = self.predict(X)
        var = (self.transformer_.scale_**2
               * self.regressor_.predict_mean_var(X))[1]
        return y, var

    def predicts(self, X, **predicts_params):
        check_is_fitted(self)
        # TODO Note that we don't have the same resize/squeeze stuff in place
        # here as in
        # https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/compose/_target.py#L275
        ys = self.regressor_.predicts(X, **predicts_params)
        return self.transformer_.inverse_transform(ys)

    def predict_distribution(self, X, **predict_distribution_params):
        def pdf(y):
            # TODO Is this correct? We won't use it before we checked
            jacobian_factor = self.transformer_.scale_
            y_trans = self.transformer_.inverse_transform(y)
            p = self.regressor_.predict_distribution(X, **predict_distribution_params)(y_trans) * jacobian_factor
            return self.transformer_.inverse_transform(p)

        return pdf


class Pipeline(pipeline.Pipeline):
    """"
    Adds (and passes through the pipeline to its final step)
    ``predict_mean_var``, ``predicts``, ``predict_distribution``.
    """
    def predict_mean_var(self, X, **predict_mean_var_params):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][1].predict_mean_var(Xt,
                                                  **predict_mean_var_params)

    def predicts(self, X, **predicts_params):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][1].predicts(Xt, **predicts_params)

    def predict_distribution(self, X, **predict_distribution_params):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][1].predict_distribution(
            Xt, **predict_distribution_params)


class Experiment(abc.ABC):
    """
    Attributes
    ----------
    learner_ : object
        After fitting, a reference to the fitted learner central to this
        experiment (useful for extracting the population, its size, fitness
        values etc.).
    """
    @property
    @abc.abstractmethod
    def algorithm(self):
        """
        The algorithm that this experiment uses for learning.
        """
        pass

    def __init__(self,
                 module,
                 seed,
                 data_seed,
                 standardize,
                 show,
                 run_name=None,
                 tracking_uri="mlflow/"):
        self.experiment_name = experiment_name(self.algorithm, module)

        self.seed = seed

        exp_path = f"experiments.{self.experiment_name}"
        exp_mod = importlib.import_module(exp_path)
        self.params = exp_mod.params

        self.data_seed = data_seed
        self.data = get_data(exp_mod.task, data_seed)

        self.standardize = standardize
        self.show = show
        self.run_name = run_name

        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)

    def run(self, **kwargs):
        for key in kwargs:
            maybe_override(self.params, key, kwargs[key])

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.run_name) as run:
            print(f"Started experiment: {self.experiment_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"Run name: {self.run_name}")
            print(f"Seed: {self.seed}")
            print(f"Data seed: {self.data_seed}")

            X = self.data["X"]
            y = self.data["y"]
            X_test = self.data["X_test"]
            y_test_true = self.data["y_test_true"]
            X_denoised = self.data["X_denoised"]
            y_denoised = self.data["y_denoised"]

            mlflow.log_param("data.seed", self.data_seed)
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
                self.estimator = Pipeline([
                    ("standardscaler", StandardScaler()),
                    ("standardscaledtargetregressor",
                     StandardScaledTargetRegressor(regressor=self.estimator))
                ])
            else:
                self.estimator = Pipeline([
                    ("identitytargetregressor",
                     IdentityTargetRegressor(regressor=self.estimator))
                ])

            self.estimator = self.estimator.fit(X, y)

            self.learner_ = self.estimator[-1].regressor_

            # TODO Consider to provide learner_ to evaluate for clarity
            self.evaluate(X, y, X_test, y_test_true, X_denoised, y_denoised)

    @abc.abstractmethod
    def init_estimator(self):
        # TODO Consider providing seed to init_estimator for clarity
        # Use self.seed here!
        self.estimator = ...

    @abc.abstractmethod
    def evaluate(self, X, y, X_test, y_test_true, X_denoised, y_denoised):
        # TODO Consider to provide learner_ to evaluate for clarity
        ...
