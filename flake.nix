{
  description = "berbl-exp";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.overlays.url = "github:dpaetzel/overlays";
  inputs.berbl = {
    type = "path";
    path = "/home/david/Projekte/berbl/berbl";
  };
  # TODO inputs.berbl.url = "github:dpaetzel/berbl";
  inputs.xcsf = {
    type = "git";
    url = "https://github.com/dpaetzel/xcsf";
    ref = "unordered-bound-intervals";
    # rev = "f5888fd00ddece4d4d9f104dcec0a7e64e584e76";
    # allRefs = true;
    submodules = true;
  };
  # TODO Add berbl-eval to reduce plotting duplication

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
