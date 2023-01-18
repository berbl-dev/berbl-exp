import warnings

import click  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mlflow  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sklearn.compose as compose  # type: ignore
import sklearn.pipeline as pipeline  # type: ignore
import tomli
from berbl import BERBL
from berbl.search.operators.metameric import DefaultToolbox
from berbl.utils import log_arrays, override_defaults
from sklearn import metrics  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from tqdm import tqdm  # type: ignore


def log_params_rec(conf, prefix=None):
    """
    Logs the given given `dict` using mlflow gracefully (recursively) handling
    further nested dicts by building dot-notation parameter names.
    """
    for key, val in conf.items():
        if type(val) is dict:
            if prefix is None:
                log_params_rec(val, prefix=key)
            else:
                log_params_rec(val, prefix=f"{prefix}.{key}")
        else:
            if prefix is None:
                mlflow.log_param(f"{key}", val)
            else:
                mlflow.log_param(f"{prefix}.{key}", val)


def callback(ga):
    """
    Prints some additional stats in each search iteration.
    """
    tqdm.write(f"Current elitist has length {str(len(ga.elitist_[0]))}.")

    lens = [len(i) for i in ga.pop_]
    dist = pd.DataFrame(np.array(np.unique(lens, return_counts=True)).T, columns=["K", "count"]).set_index("K")
    tqdm.write(f"\nLength distribution:\n {dist}.")

    try:
        lens_ = ga.lens_
        tqdm.write(f"\nLast used niches during selection: {ga.lens_}.")
    except:
        tqdm.write("Not using metameric GA, so not printing used niches.")



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


class StandardScaledTargetRegressor(compose.TransformedTargetRegressor):
    """
    A `TransformedTargetRegressor` that uses `transformer=StandardScaler`.

    Adds (and passes through the pipeline to its final step) `predict_mean_var`,
    `predicts`, `predict_distribution`. Note that `predict_mean_var` and
    `predict_distribution` are different for different transformers which is why
    `StandardScaler` is fixed here.
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
        # Note that we don't have the same resize/squeeze stuff in place
        # here as in
        # https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/compose/_target.py#L275
        ys = self.regressor_.predicts(X, **predicts_params)
        return np.array([self.transformer_.inverse_transform(y) for y in ys])

    def predict_distribution(self, X, **predict_distribution_params):

        def pdf(y):
            raise NotImplementedError(
                "I'm not yet 100% sure whether this is correct")
            jacobian_factor = self.transformer_.scale_
            y_trans = self.transformer_.inverse_transform(y)
            p = self.regressor_.predict_distribution(
                X, **predict_distribution_params)(y_trans) * jacobian_factor
            return self.transformer_.inverse_transform(p)

        return pdf


def maybe_override(params: dict, param: str, value: object) -> None:
    """
    Overrides the entry at position `param` in the `params` dictionary with
    value `value`. Prints warnings to stdout so the user knows what is going on.
    """
    if value is None:
        return
    elif param in params and params[param] != value:
        warnings.warn(f"Warning: Overriding {param}={params[param]} with "
                      f"{param}={value}")
        params[param] = value
    elif param not in params:
        warnings.warn(f"Warning: Forcing {param}={value}")
        params[param] = value


@click.group()
def cli():
    pass


@cli.command()
@click.option("-s",
              "--seed",
              type=click.IntRange(min=0),
              default=0,
              show_default=True,
              help="Seed to use for initializing RNGs")
@click.option(
    "-F",
    "--config-file",
    default=None,
    type=str,
    show_default=True,
    # TODO Write help
    help="Algorithm configuration to use")
@click.option("--n-iter",
              default=None,
              type=int,
              show_default=True,
              help="Number of iterations to run the metaheuristic for")
@click.option(
    "--match",
    default=None,
    type=str,
    show_default=True,
    help=("Matching function family to use, one of "
          "\"radial\" (1D-only), \"softint\" (1D-only), \"interval\""))
@click.option("--run-name", type=str, default=None)
@click.option("--tracking-uri", type=str, default="mlruns")
@click.option("--experiment-name", type=str, required=True)
@click.argument("NPZFILE")
def run(seed, config_file, n_iter, match, run_name, tracking_uri,
        experiment_name, npzfile):
    """
    Run BERBL on the given DATA, logging results using mlflow under the given
    EXPERIMENT_NAME.
    """

    print(f"Logging to mlflow tracking URI {tracking_uri}.")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Setting experiment name to \"{experiment_name}\".")
    mlflow.set_experiment(experiment_name)

    print(f"Setting run name to \"{run_name}\".")
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Run ID is {run.info.run_id}.")

        # TODO Log data hash (i.e. npzfile hash)

        print(f"RNG seed is {seed}.")
        random_state = check_random_state(seed)
        mlflow.log_param("seed", seed)

        data = np.load(npzfile)
        mlflow.log_param("data.fname", npzfile)

        X = data["X"]
        y = data["y"]
        X_test = data["X_test"]
        y_test_true = data["y_test_true"]

        # See arguments of metameric.DefaultToolbox.
        # TODO Hardcoding this here ain't nice
        config_default = dict(
            matchcls=None,
            match=None,
            search=None,
            toolbox=dict(init=None,
                         fitness=None,
                         select=None,
                         mutate=None,
                         crossover=None),
        )

        if config_file is not None:
            # Load config.
            with open(config_file, "rb") as f:
                config = tomli.load(f)

            config = override_defaults(config, config_default, "config")
        else:
            config = config_default

        # Override parameters if specified as CLI arguments.
        maybe_override(config, "matchcls", match)
        # TODO maybe_override(config["search"], "pop_size", pop_size)
        maybe_override(config["search"], "n_iter", n_iter)
        maybe_override(config["search"], "callback_gen", callback)

        # Parameters are final now and we can log them.
        log_params_rec(config)

        toolbox = DefaultToolbox(
            matchcls=config["matchcls"],
            random_state=random_state,
            params_match=config["match"],
            params_init=config["toolbox"]["init"],
            params_fitness=config["toolbox"]["fitness"],
            params_select=config["toolbox"]["select"],
            params_mutate=config["toolbox"]["mutate"],
            params_crossover=config["toolbox"]["crossover"],
        )
        # TODO FIXME
        # Provide all parameters as keyword arguments.
        # This way, we can override Rule, Mixing and
        # Mixture hyperparameters.
        # **low_params,

        pipe = Pipeline([
            ("standardscaler", StandardScaler()),
            ("standardscaledtargetregressor",
             StandardScaledTargetRegressor(regressor=BERBL(
                 toolbox=toolbox, params_search=config["search"])))
        ])

        with np.errstate(all="ignore"):
            estimator = pipe.fit(X, y)
            y_test_pred, y_test_pred_var = estimator.predict_mean_var(X_test)
            y_test_cls = estimator.predicts(X_test)

        log_arrays("results",
                   y_test_pred=y_test_pred,
                   y_test_pred_var=y_test_pred_var,
                   y_test_cls=y_test_cls)

        # Log additional statistics to maybe better gauge solution performance.
        mae = metrics.mean_absolute_error(y_test_true, y_test_pred)
        mse = metrics.mean_squared_error(y_test_true, y_test_pred)
        # TODO Shouldn't R2 be computed in-sample?
        r2 = metrics.r2_score(y_test_true, y_test_pred)
        mlflow.log_metric("elitist.size",
                          estimator[-1].regressor_.search_.size_[0], n_iter)
        mlflow.log_metric("elitist.ln_p_M_D",
                          estimator[-1].regressor_.search_.ln_p_M_D_[0],
                          n_iter)
        mlflow.log_metric("elitist.mae", mae, n_iter)
        mlflow.log_metric("elitist.mse", mse, n_iter)
        mlflow.log_metric("elitist.r2_score", r2, n_iter)

        # TODO Log elitist, final population etc.

        if False:
            # We do not log plots here but rather generate plots from mlflow data.
            fig, ax = plt.subplots(1)
            ax.scatter(X_test, y_test_true, marker="+", color="C0")
            ax.plot(X_test, y_test_pred, color="C1")
            matchs = estimator[1].regressor_.search_.elitist_[0]
            X_test_scaled = pipe[0].transform(X_test)
            X_scaled = pipe[0].transform(X)
            for match in matchs:
                m = match.match(X_test_scaled)
                # Add jitter for visibility.
                # m += np.random.normal(loc=0, scale=0.01, size=m.shape)
                ax.plot(X_test, m, linestyle="dotted")
            # for i, match in enumerate(matchs):
            #     ax.hlines(y_test_true.min() - 0.1 * (i + 1), match.l, match.u, linestyle="dotted")
            plt.show()


if __name__ == "__main__":
    cli()
