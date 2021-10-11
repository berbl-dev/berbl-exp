let
  bootstrap = import <nixpkgs> { };
  pkgs = import (bootstrap.fetchFromGitHub {
    owner = "NixOS";
    repo = "nixpkgs";
    rev = "f18bd6f9192e0f4b78419efe4caaf0fd37a0a346";
    sha256 = "0d0m25xxl9s6ljk27x9308zqawdbmhd7m9cwkcajsbn684imcw85";
  }) { };
in
with pkgs; callPackage ./default.nix {
  inherit lib stdenv fetchgit cmake;
  toPythonModule = python3Packages.toPythonModule;
  python = python3;
}
