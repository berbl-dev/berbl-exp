{
  description = "berbl-exp";

  inputs.berbl.url = "github:berbl-dev/berbl/v0.1.0-beta";

  inputs.xcsf = {
    type = "git";
    url = "https://github.com/dpaetzel/xcsf";
    ref = "unordered-bound-intervals";
    # rev = "f5888fd00ddece4d4d9f104dcec0a7e64e584e76";
    # allRefs = true;
    submodules = true;
  };
  inputs.xcsf.inputs.nixpkgs.follows = "berbl/nixpkgs";

  outputs = { self, berbl, xcsf }: {

    defaultPackage.x86_64-linux = with import berbl.inputs.nixpkgs {
      system = "x86_64-linux";
      overlays = with berbl.inputs.overlays.overlays; [ mlflow ];
    };
      let python = python39;
      in python.pkgs.buildPythonPackage rec {
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
          xcsf.defaultPackage.x86_64-linux
        ];

        doCheck = false;

        meta = with lib; {
          description =
            "Library for performing experiments with the BERBL library";
          license = licenses.gpl3;
        };
      };
  };
}
