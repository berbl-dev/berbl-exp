from .. import default_xcs_params

task = "book.sine"

params = default_xcs_params() | {
    "MAX_TRIALS" : 100000,
    # From parameter study.
    "POP_SIZE": 200,
    "E0": 0.05,
    "BETA": 0.01,
}
