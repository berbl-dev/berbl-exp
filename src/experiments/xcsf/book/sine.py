from .. import default_xcs_params

task = "book.sine"

# TODO Use proper parameter settings
params = default_xcs_params() | {
    "MAX_TRIALS" : 100000,
}
