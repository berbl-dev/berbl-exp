from evaluation import *


node = "oc-compute03"

cl = mlflow.tracking.MlflowClient(
    tracking_uri=f"mlruns-remote-{node}/localized/mlruns",
    registry_uri=f"mlruns-remote-{node}/localized/mlruns")


# Get list of run names.
import run as runpy


berbl_experiment_names = [f"berbl.{exp}" for exp in runpy.berbl_experiments]
xcsf_experiment_names = [f"xcsf.{exp}" for exp in runpy.xcsf_experiments]

# Only consider runs that ran from these commits.
shas = [
    "0dc54e4303bba8e187d622bdb0b83b63f9a74998",
    "a15a5b30a5de093abf296fb80b4fb3add23598da"
]


# Get all run data gathered for berbl and perform a few checks.
berbl_experiments = {}
for exp in berbl_experiment_names:
    berbl_experiments[exp] = flatten(
        [git_runs(cl, exp, sha, unfinished=True) for sha in shas])
    # 25 runs w/ and 25 runs w/o standardization
    assert len(berbl_experiments[exp]) == 50, (
        f"There should be 50 runs for {exp} "
        f"but there are {len(berbl_experiments[exp])}.")
    # Check whether all runs being considered are finished
    assert len([
        e for e in berbl_experiments[exp] if e.info.status != "FINISHED"
    ]) == 0, f"Some runs for {exp} are not FINISHED yet."

# Performance analysis: Compare with Drugowitsch's results (p(M | D)).

# Do not concern ourselves yet with the standardized experiments but only with
# the 1:1 reproduction of Drugowitsch's results.
repr_berbl_experiments = {
    exp_name: [
        r for r in berbl_experiments[exp_name]
        # Slightly weird that Booleans are stored as strings.
        if r.data.params["standardize"] == "False"
    ]
    for exp_name in berbl_experiments
}
# Introduce a shorthand for the REPL.
rpe = repr_berbl_experiments


df_repr_berbl_experiments = apply(df, repr_berbl_experiments)

# Update shorthand.
dfrpe = df_repr_berbl_experiments

# These are Drugowitsch's results on these tasks (taken from his book).
drugowitsch_ga = pd.DataFrame({
    "generated_function": {
        "elitist.p_M_D": 118.81,
        "elitist.size": 2
    },
    "sparse_noisy_data": {
        "elitist.p_M_D": -159.07,
        "elitist.size": 2
    },
    "variable_noise": {
        "elitist.p_M_D": -63.12,
        "elitist.size": 2
    },
    "sine": {
        "elitist.p_M_D": -155.68,
        "elitist.size": 7
    },
})
drugowitsch_mcmc = pd.DataFrame({
    "generated_function": {
        "elitist.p_M_D": 174.50,
        "elitist.size": 3
    },
    "sparse_noisy_data": {
        "elitist.p_M_D": -158.55,
        "elitist.size": 2
    },
    "variable_noise": {
        "elitist.p_M_D": -58.59,
        "elitist.size": 2
    },
    "sine": {
        "elitist.p_M_D": -29.39,
        "elitist.size": 5
    },
})


def task_name(exp_name):
    return re.sub(".*\..*\.", "", exp_name)


# for each *task* add one table that puts together book vs modular vs drugowitsch vs xcsf
metrics = ["elitist.size", "elitist.mse", "elitist.p_M_D"]
for exp, df in sorted(df_repr_berbl_experiments.items(),
                      key=lambda it: task_name(it[0])):

    tname = task_name(exp)
    print()
    print(f"# {exp}")
    print()

    d = pd.DataFrame({
        "mean": df[metrics].mean(),
        "std": df[metrics].std(),
        "max": df[metrics].max(),
        "min": df[metrics].min()
    })
    d = pd.concat([d, drugowitsch_ga[tname]], axis=1)
    d = d.rename(columns={tname: "Drugowitsch's GA"})
    d = pd.concat([d, drugowitsch_mcmc[tname]], axis=1)
    d = d.rename(columns={tname: "Drugowitsch's MCMC"})
    d = d.T

    d2 = df.groupby("data.seed")[metrics].mean()
    d2.index = [f"data seed {i}: mean" for i in d2.index]
    d = pd.concat([d, d2])

    d2 = df.groupby("data.seed")[metrics].std()
    d2.index = [f"data seed {i}: std" for i in d2.index]
    d = pd.concat([d, d2])

    print(d.to_markdown())
