let
  bootstrap = import <nixpkgs> { };
  pkgs = import (bootstrap.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "35acd3ed181ee1622da68ec26e0b516d7e585837";
    sha256 = "1fswzd55nx3drdr48qv3by49xkdw8rypp2wdvcciyda3plxydhpk";
  }) {
    config.packageOverrides = super: {
      python3 = super.python3.override {
        packageOverrides = python-self: python-super: {
          pandas = python-super.pandas.overrideAttrs (attrs: rec {
            pname = "pandas";
            version = "1.3.4";

            src = python-super.fetchPypi {
              inherit pname version;
              sha256 = "1z3gm521wpm3j13rwhlb4f2x0645zvxkgxij37i3imdpy39iiam2";
            };
          });
          sqlalchemy = python-super.sqlalchemy.overrideAttrs (attrs: rec {
            pname = "SQLAlchemy";
            version = "1.3.13";
            src = python-super.fetchPypi {
              inherit pname version;
              sha256 =
                "sha256:1yxlswgb3h15ra8849vx2a4kp80jza9hk0lngs026r6v8qcbg9v4";
            };
            doInstallCheck = false;
          });
          alembic = python-super.alembic.overrideAttrs (attrs: rec {
            pname = "alembic";
            version = "1.4.1";
            src = python-super.fetchPypi {
              inherit pname version;
              sha256 =
                "sha256:0a4hzn76csgbf1px4f5vfm256byvjrqkgi9869nkcjrwjn35c6kr";
            };
            propagatedBuildInputs = with python-super; [
              python-editor
              python-dateutil
              python-self.sqlalchemy
              Mako
            ];
            doInstallCheck = false;
          });
          mlflowPatched = (python-super.mlflow.override {
            sqlalchemy = python-self.sqlalchemy;
            # requires an older version of alembic
            alembic = python-self.alembic;
          }).overrideAttrs (attrs: {
            propagatedBuildInputs = attrs.propagatedBuildInputs
              ++ (with python-self; [
                importlib-metadata
                prometheus-flask-exporter
                azure-storage-blob
              ]);
            meta.broken = false;
          });
          # berbl = with pkgs; with python-super; pkgs.callPackage ./berbl/default.nix {
          baycomp = with pkgs;
            with python-super;
            import ./deps/baycomp/default.nix {
              inherit lib matplotlib scipy;
              buildPythonPackage = buildPythonPackage;
            };
          berbl = with pkgs;
            with python-super;
            import ./berbl/default.nix {
              inherit lib deap numpy numpydoc scipy scikitlearn sphinx
                hypothesis pytest;
              pandas = python-self.pandas;
              mlflow = python-self.mlflowPatched;
              buildPythonPackage = buildPythonPackage;
            };
          xcsf = with super.pkgs;
            callPackage ./deps/xcsf/default.nix {
              inherit lib stdenv fetchgit cmake;
              toPythonModule = python3Packages.toPythonModule;
              python = python3;
              tqdm = python3Packages.tqdm;
            };
        };
      };
    };
  };
  python = (pkgs.python3.withPackages (ps:
    with ps; [
      baycomp
      click
      deap
      ipython
      mlflowPatched
      networkx
      numpy
      pandas
      berbl
      pygraphviz
      scipy
      scikitlearn
      seaborn
      tabulate # for pandas markdown support
      xcsf
    ]));
in pkgs.mkShell rec {
  venvDir = ".venv";
  packages = with pkgs; [ pkgs.graphviz python3Packages.venvShellHook python ];
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install pystan==2.19.1.1
  '';
  # NOTE We first search pure Nix Python packages and then the venv. Otherwise
  # the venv numpy is used which seems to not be able to find zlib.
  postShellHook = ''
    export PYTHONPATH=${python}/${python.sitePackages}:$PWD/${venvDir}/${python.sitePackages}/:$PYTHONPATH
  '';
}
