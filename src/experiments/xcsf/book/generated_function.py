from .. import default_xcs_params

task = "book.generated_function"

params = default_xcs_params() | {
    "MAX_TRIALS" : 100000,
    # From parameter study.
    "POP_SIZE": 200,
    "E0": 0.01,
    "BETA": 0.01,
}
