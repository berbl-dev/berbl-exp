let
  bootstrap = import <nixpkgs> { };
  pkgs = import (bootstrap.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "f18bd6f9192e0f4b78419efe4caaf0fd37a0a346";
    sha256 = "0d0m25xxl9s6ljk27x9308zqawdbmhd7m9cwkcajsbn684imcw85";
  }) {
    config.packageOverrides = super: {
      python3 = super.python3.override {
        packageOverrides = python-self: python-super: {
          pyparsing3 = python-super.buildPythonPackage rec {
            # Copied from nixpkgs and slightly adjusted for 3.0.0rc2.
            pname = "pyparsing";
            version = "3.0.0rc2";

            src = super.fetchFromGitHub {
              owner = "pyparsing";
              repo = pname;
              rev = "pyparsing_${version}";
              sha256 =
                "sha256:0j2y4075s9596826cjw5k3z7c7fdnn53nkdahn466h9l0zgbyfx1";
            };

            # https://github.com/pyparsing/pyparsing/blob/847af590154743bae61a32c3dc1a6c2a19009f42/tox.ini#L6
            checkInputs = [ python-super.coverage ];
            checkPhase = ''
                # coverage run --branch simple_unit_tests.py
                # coverage run --branch unitTests.py
              '';

            meta = with super.lib; {
              homepage = "https://github.com/pyparsing/pyparsing";
              description =
                "An alternative approach to creating and executing simple grammars, vs. the traditional lex/yacc approach, or the use of regular expressions";
              license = licenses.mit;
              maintainers = with maintainers; [ kamadorueda ];
            };
          };
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
          prolcs = pkgs.callPackage ./prolcs/default.nix {
            buildPythonPackage = pkgs.python3Packages.buildPythonPackage;
          };
          xcsf = with pkgs; callPackage ./xcsf/default.nix {
            inherit lib stdenv fetchgit cmake;
            toPythonModule = python3Packages.toPythonModule;
            python = python3;
            tqdm = python3Packages.tqdm;
          };
        };
      };
    };
  };
  env = pkgs.python3.withPackages (ps:
    with ps; [
      click
      deap
      ipython
      mlflowPatched
      numpy
      pandas
      prolcs
      pyparsing3
      scipy
      scikitlearn
      seaborn
      xcsf
    ]);
in
pkgs.mkShell {
  packages = [
    env
    pkgs.python3Packages.mlflowPatched
  ];
}
