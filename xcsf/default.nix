{ stdenv, toPythonModule, cmake, lib, fetchgit, fetchzip, python, tqdm }:


toPythonModule (stdenv.mkDerivation rec {
  pname = "xcsf";
  version = "1.1.1";

  src = ./xcsf-v.1.1.2-incl-submodules.tar.gz;
  # src = /home/david/Code/xcsf;

  nativeBuildInputs = [ cmake ];

  buildInputs = [
    python
  ];
  # TODO Add openmp?

  propagatedBuildInputs = [
    tqdm
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RELEASE"

    "-DXCSF_PYLIB=ON"
    "-DENABLE_TESTS=ON"
    "-DPARALLEL=ON"
  ];

  installPhase = ''
    mkdir -p $out/${python.sitePackages}
    cp -r xcsf $out/${python.sitePackages}
  '';

  meta = with lib; {
    description = "Implementation of the XCSF learning classifier system";
    longDescription = ''
      Preen's Python bindings for his implementation of the XCSF learning
      classifier system in C.
    '';
    homepage = "https://github.com/rpreen/xcsf";
    license = licenses.gpl3;
    maintainers = [ maintainers.dpaetzel ];
    platforms = platforms.all;
  };
})
