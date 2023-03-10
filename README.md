# visualizeMSLD: Perform Structural Analysis of MSLD Trajectories

A simple set of functions to extract physical frames from and visualize Multi-Site Lambda Dynamics (MSLD) trajectories generated from Adaptive Landscape Flattening (ALF)<sup>1</sup>.  
  
 These functions take in many system specific variables generated by [MSLD-py-prep]("https://github.com/Vilseck-Lab/msld-py-prep/README.md")<sup>2</sup>, as well as lambda trajectory files (typically a .dat file in the ALF production analysis directory), and a .dcd MSLD trajectory file generated by CHARMM. 
  
  The output is either a PyMOL visualization of the lambda trajectory (using the transparency setting) via the `visualize_pymol` function, or, a directory called run{i}_frames containing subdirectories of each physical ligand's frames throughout the trajectory via the `grab_frames` function. Both of these functions use the output of `get_subs_on.`

These scripts can be run from within the ALF directory where the trajectory to be visualized was created. 

<sup>1</sup>[_J. Phys. Chem. B._  2017, 121, 15, 3626–3635]("https://doi.org/10.1021/acs.jpcb.6b09656")
  
<sup>2</sup>[_J. Chem. Inf. Model._ 2022, 62, 6, 1479–1488]("https://doi.org/10.1021/acs.jcim.2c00047")

## Dependencies
- Python 3 or later versions
- PyMOL
- pyCHARMM

## Authorship
Luis F. Cervantes and Furyal Ahmed
