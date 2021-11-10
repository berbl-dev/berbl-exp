from .. import default_xcs_params

# TODO Use properly optimized parameter settings
task = "book.generated_function"
params = default_xcs_params() | {
    "MAX_TRIALS" : 100000,
    "POP_SIZE": 50, # “10 times the expected number of rules”
    "ALPHA": 1, # typical value in literature (stein2019, stalph2012c)
    "M_PROBATION" : int(1e8), # quasi disabled
    "BETA": 0.005, # lower value required if high noise
    "THETA_SUB":
    20,  # not sensitive, typical value in literature (stein2019, stalph2012c)
    "SET_SUBSUMPTION": True, # no compact solutions if no subsumption
    "EA_SELECT_TYPE": "tournament",  # tournament is the de-facto standard
}
