from .. import default_xcs_params

# TODO Use properly optimized parameter settings
task = "book.generated_function"

params = default_xcs_params() | {
    "MAX_TRIALS" : 100000,
}
