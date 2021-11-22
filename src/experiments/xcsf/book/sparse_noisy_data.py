from .. import default_xcs_params

task = "book.sparse_noisy_data"

params = default_xcs_params() | {
    "MAX_TRIALS" : 100000,
    # From parameter study.
    "POP_SIZE": 100,
    "E0": 0.001,
    "BETA": 0.005,
}
