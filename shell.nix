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
          pandas = python-super.pandas.overrideAttrs (attrs: rec {
            pname = "pandas";
            version = "1.3.4";

            src = python-super.fetchPypi {
              inherit pname version;
              sha256 = "1z3gm521wpm3j13rwhlb4f2x0645zvxkgxij37i3imdpy39iiam2";
            };
          });
          baycomp = python-super.buildPythonPackage rec {
            pname = "baycomp";
            version = "unstable-8c4a22";

            src = super.fetchFromGitHub {
              owner = "janezd";
              repo = pname;
              rev = "8c4a2253e875fc1eae2b00ab9da77c14940885e2";
              sha256 =
                "sha256:1yan1kk72yci55g2k3zala2s3711bvw8r2zlq1xh0vv1pdzk2c8k";
            };

            propagatedBuildInputs = with python-super; [ matplotlib scipy ];

            meta = with super.lib; {
              homepage = "https://github.com/janezd/baycomp";
              description = "A library for Bayesian comparison of classifiers";
              license = licenses.mit;
              maintainers = with maintainers; [ dpaetzel ];
            };
          };
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
          berbl = with pkgs; with python-super; pkgs.callPackage ./berbl/default.nix {
            inherit lib deap numpy scipy scikitlearn hypothesis pytest;
            pandas = python-self.pandas;
            mlflow = python-self.mlflowPatched;
            buildPythonPackage = buildPythonPackage;
          };
          xcsf = with super.pkgs;
            callPackage ./xcsf/default.nix {
              inherit lib stdenv fetchgit cmake;
              toPythonModule = python3Packages.toPythonModule;
              python = python3;
              tqdm = python3Packages.tqdm;
            };
        };
      };
    };
  };
python =
    (pkgs.python3.withPackages (ps: with ps; [
      baycomp
      click
      deap
      ipython
      mlflowPatched
      numpy
      pandas
      berbl
      pyparsing3
      scipy
      scikitlearn
      seaborn
      xcsf
    ]));
in pkgs.mkShell rec {
  venvDir = ".venv";
  packages = with pkgs; [
    python3Packages.venvShellHook
    python
  ];
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install pystan==2.19.1.1
  '';
  postShellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$PWD/${venvDir}/${python.sitePackages}/:${python}/${python.sitePackages}:$PYTHONPATH
  '';
}
