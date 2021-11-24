# Repeating the experiments


1. Install
  [Nix](https://nixos.org/manual/nix/stable/installation/installing-binary.html)
  in order to be able to run `nix-shell` later (alternatively, check `shell.nix`
  and install the dependencies manually).
  Note that [Nix does not yet support
  Windows](https://nixos.org/manual/nix/stable/installation/supported-platforms.html).
2. Clone the repository (`git clone …`). Run the next steps from within the
   cloned repository.
3. Clone the `berbl` repository to `berbl`.
4. Run `nix-shell` to enter a shell that contains all dependencies (may take
   some time to complete).
5. Run XCSF parameter study, storing results in `mlrun-ps` (currently requires
   Slurm)
   ```bash
   PYTHONPATH=src:$PYTHONPATH python -m run paramsearch oc-compute03 --tracking-uri=mlruns-ps
   ```
6. Evaluate parameter study
   ```bash
   PYTHONPATH=src:$PYTHONPATH python -m eval-ps mlruns-ps/
   ```
   making sure that the Git commit hash is set correctly in `eval-ps` (to the
   Git commit hash that was used to run the parameter study).
7. Enter parameter settings recommended by parameter study in experiment
   configurations under `src/experiments/xcsf/book/`.
8. Run all experiments using Slurm (or adjust the script to not use Slurm)
   ```bash
   PYTHONPATH=src:$PYTHONPATH python -m run slurm HOST
   ```
   or check out
   ```bash
   PYTHONPATH=src:$PYTHONPATH python -m run --help
   ```
   for more options.
9. Evaluate runs:
   ```bash
   PYTHONPATH=src:$PYTHONPATH python -m eval mlruns/
   ```
