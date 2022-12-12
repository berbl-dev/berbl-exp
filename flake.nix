{
  description = "berbl-exp";

  inputs = {
    berbl.url = "github:berbl-dev/berbl/add-hardinterval";

    # FIXME Broken due to only supporting python39 (have to change that in the
    # flake)
    xcsf = {
      type = "git";
      # TODO Update xcsf from upstream
      url = "https://github.com/dpaetzel/xcsf";
      ref = "unordered-bound-intervals";
      # rev = "f5888fd00ddece4d4d9f104dcec0a7e64e584e76";
      # allRefs = true;
      submodules = true;
    };
    xcsf.inputs.nixpkgs.follows = "berbl/nixos-config/nixpkgs";
  };

  outputs = { self, berbl, xcsf }:

    let
      nixpkgs = berbl.inputs.nixos-config.inputs.nixpkgs;
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python310;
    in rec {

      defaultPackage.${system} = python.pkgs.buildPythonPackage rec {
        pname = "berbl-exp";
        version = "1.0.0";

        src = self;

        # We use pyproject.toml.
        format = "pyproject";

        propagatedBuildInputs = with python.pkgs; [
          berbl.defaultPackage.x86_64-linux
          click
          mlflow
          numpy
          numpydoc
          pandas
          scipy
          tomli
          xcsf.defaultPackage.${system}

          ipython
        ];

        doCheck = false;

        meta = with pkgs.lib; {
          description =
            "Library for performing experiments with the BERBL library";
          license = licenses.gpl3;
        };
      };

      devShell.${system} = pkgs.mkShell {

        buildInputs = with python.pkgs; [
          ipython
          (python.withPackages
            (p: defaultPackage.${system}.propagatedBuildInputs))
          venvShellHook
        ];

        venvDir = "./_venv";

        postShellHook = ''
          unset SOURCE_DATE_EPOCH

          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]
          }:$LD_LIBRARY_PATH";
        '';

        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
        '';

      };
    };
}
