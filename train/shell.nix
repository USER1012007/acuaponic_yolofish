{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [ 
    pkgs.glib 
  ];

  shellHook = ''
    # Exportamos la librería
    export LD_LIBRARY_PATH="${pkgs.glib.out}/lib:$LD_LIBRARY_PATH"
    
    echo "Entorno de Hailo SDK cargado correctamente."
  '';
}
