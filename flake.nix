{
  description = "berbl-exp";

  # 2022-01-24
  inputs.nixpkgs.url =
    "github:NixOS/nixpkgs/8ca77a63599ed951d6a2d244c1d62092776a3fe1";

  inputs.overlays.url = "github:dpaetzel/overlays";

  inputs.berbl.url = "github:berbl-dev/berbl";

  inputs.xcsf = {
    type = "git";
    url = "https://github.com/dpaetzel/xcsf";
    ref = "unordered-bound-intervals";
    # rev = "f5888fd00ddece4d4d9f104dcec0a7e64e584e76";
    # allRefs = true;
    submodules = true;
  };

  outputs = { self, nixpkgs, overlays, berbl, xcsf }: {

    defaultPackage.x86_64-linux = with import nixpkgs {
      system = "x86_64-linux";
      overlays = with overlays.overlays; [ mlflow ];
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
          sphinx
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
