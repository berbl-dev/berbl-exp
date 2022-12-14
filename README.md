# Experiment automation for BERBL


Submit jobs like this:

1. Prebuild the environment on the node (typically only has to be done if one of
   `flake.nix` or `flake.lock` changes).

   ```
   srun --partition=cpu-prio --nodelist=oc-compute03 --mem=3G nix develop --command python --version
   ```

2. Enter the development shell to get the necessary packages to run the
   submission script.

    ```
    nix develop
    ```

3. Submit a repeated experiment.

    ```
    python scripts/submit.py submit --n-reps=2 -F configs/sine.toml --experiment-name="test.sine" tasks/gecco-2022/berbl-book-sine-0.npz
    ```

4. Watch the queue

   ```
   python scripts/shelp.py watch
   ```
