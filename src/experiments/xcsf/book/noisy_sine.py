from .. import default_xcs_params

task = "book.noisy_sine"

# TODO Use proper parameter settings
params = default_xcs_params() | {
    "MAX_TRIALS" : 10000
}
