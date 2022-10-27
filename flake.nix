{
  description = "berbl-exp";

  inputs = {
    nixos-config.url = "github:dpaetzel/nixos-config";

    # inputs.berbl.url = "github:berbl-dev/berbl/v0.1.0-beta";
    berbl.url = "github:berbl-dev/berbl/develop";
    berbl.inputs.nixpkgs.follows = "nixos-config/nixpkgs";

    xcsf = {
      type = "git";
      # TODO Update xcsf from upstream
      url = "https://github.com/dpaetzel/xcsf";
      ref = "unordered-bound-intervals";
      # rev = "f5888fd00ddece4d4d9f104dcec0a7e64e584e76";
      # allRefs = true;
      submodules = true;
    };
    xcsf.inputs.nixpkgs.follows = "nixos-config/nixpkgs";
  };

  outputs = { self, berbl, xcsf, nixos-config }:

    let
      nixpkgs = nixos-config.inputs.nixpkgs;
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = with berbl.inputs.overlays.overlays; [ mlflow ];
      };
      # TODO Upgrade this (and berbl etc.) to python310
      python = pkgs.python39;
    in rec {
      defaultPackage.${system} = python.pkgs.buildPythonPackage rec {
        pname = "berbl-exp";
        version = "1.0.0";

        src = self;

        # We use pyproject.toml.
        format = "pyproject";

        propagatedBuildInputs = with python.pkgs; [
          berbl.defaultPackage.x86_64-linux
          mlflow
          numpy
          numpydoc
          pandas
          scipy
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
    };
}
