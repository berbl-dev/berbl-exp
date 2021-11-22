from .. import default_xcs_params

task = "book.sparse_noisy_data"

# TODO Use proper parameter settings
params = default_xcs_params() | {
    "MAX_TRIALS" : 100000,
}
