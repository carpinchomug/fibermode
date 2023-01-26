{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

      in
      {
        packages.default = with pkgs.python3Packages; buildPythonPackage {
          pname = "fibermode";
          version = "0.1.0";
          src = ./.;
          format = "pyproject";
          nativeBuildInputs = [ setuptools ];
          propagatedBuildInputs = [ numpy scipy ];
        };

        devShells.default = with pkgs; mkShell {
          packages = [
            (python3.withPackages (ps: with ps; [
              numpy
              scipy
              setuptools
              python-lsp-server
              python-lsp-server.optional-dependencies.all
              python-lsp-black
              pyls-isort
            ]))
          ];
        };
      }
    );
}
