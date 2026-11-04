{
  description = "typst development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    {
      nixpkgs,
      ...
    }:
    let
      systems = nixpkgs.lib.platforms.unix;
      eachSystem =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f (
            import nixpkgs {
              inherit system;
              config = { };
              overlays = [ ];
            }
          )
        );
      info = fromTOML (builtins.readFile ./typst.toml);
    in
    {
      packages = eachSystem (
        pkgs:
        let
          inherit (pkgs.lib.fileset) toSource unions;
          mltyp = pkgs.buildTypstPackage {
            pname = info.package.name;
            inherit (info.package) version;
            src = toSource {
              root = ./.;
              fileset = unions [
                ./src
                ./typst.toml
              ];
            };
          };
        in
        {
          inherit mltyp;
          default = mltyp;
        }
      );

      devShells = eachSystem (pkgs: {
        default = pkgs.mkShellNoCC {
          packages = with pkgs; [
            typst
            typstyle
          ];
        };
      });
    };
}
