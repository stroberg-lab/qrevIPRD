mol delete top
mol load xyz boxsize_5_5_5/run_bulk_nL2_nC1/trajectory_0/LigandDiffusion_out_bulk.h5.xyz
mol delrep 0 top
display resetview
mol representation VDW 0.700000 16.0
mol selection name type_0
mol color ColorID 0
mol addrep top
mol representation VDW 0.700000 16.0
mol selection name type_1
mol color ColorID 1
mol addrep top
mol representation VDW 0.881945 16.0
mol selection name type_2
mol color ColorID 2
mol addrep top
animate goto 0
color Display Background white
molinfo top set {center_matrix} {{{1 0 0 0}{0 1 0 0}{0 0 1 0}{0 0 0 1}}}
