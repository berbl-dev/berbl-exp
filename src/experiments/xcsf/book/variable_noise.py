from .. import default_xcs_params

task = "book.variable_noise"

# TODO Use proper parameter settings
params = default_xcs_params() | {
    "MAX_TRIALS" : 10000
}
