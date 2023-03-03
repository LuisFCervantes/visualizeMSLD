# Set of functions to visualize MSLD trajectories
# Written by: Luis F Cervantes and Furyal Ahmed (2/23)

import numpy as np
import sys
import string
import os
from functools import reduce
import pycharmm
import pycharmm.lingo as lingo
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.select as select
import pycharmm.shake as shake
import pycharmm.charmm_file as charmm_file

from pycharmm.lib import charmm as libcharmm



def take_overlap(*input):
    """
    Given an unpacked list `input` of np.arrays,
    take the overlap of the arrays given their first
    column. 

    In:
    `input` : unpacked list of np.arrays

    Out:
    `result`: list of np.arrays
    """
    n = len(input)
    maxIndex = max(array[:, 0].max() for array in input)
    indicator = np.zeros(maxIndex + 1, dtype=int)
    for array in input:
        indicator[array[:, 0]] += 1
    indicator = indicator == n

    result = []
    for array in input:
        # Look up each integer in the indicator array
        mask1 = indicator[array[:, 0]]
        # Use boolean indexing to get the sub array
        result.append(array[mask1])

    return result


def get_subs_on(LambdaFile, nsubs, cutoff=0.99, nsavc=10000, nsavl=10):
    """
    Create subs_on matrix of shape (nframes x nsites), where 
    A_{i,j} is the substituent index that is on at the ith frame 
    and jth site (zero-based indexing)

    In:
        LambdaFile           (str) : path to lambda trajectory file
        nsubs       [int, int,...] : list of nsubs per site
        cutoff             (float) : lambda cutoff
        nsavc                (int) : freq of saving frames
        nsavl                (int) : freq of saving lambdas

    Out:
        subs_on      2-D np.array  : subs_on matrix
    """
    # Check to make sure frames align with lambda values
    assert (nsavc/nsavl).is_integer, f"Frequency of saving lambdas and frames are not multiples of each other"
    
    # Define sites and skip
    nsites = len(nsubs)
    skip = int(nsavc/nsavl)

    # Load lambda trajectory
    lams = np.loadtxt(LambdaFile)

    # Extract physical substituents based on cutoff
    physical_subs = []
    lambdasPerFrame = []
    for site in range(nsites):
        # Find indices for all subs at one site
        if site == 0:
            index1 = 0
        else:
            index1 = np.cumsum(nsubs[:site+1])[site-1]
    
        index2 = np.cumsum(nsubs[:site+1])[-1] -1  

        # Get lambdas for the site based on indices 
        lams_site = lams[skip-1::skip,index1:index2+1]
        lambdasPerFrame.append(lams_site)

        # Retrive physical substituents on a site
        mask = lams_site >= cutoff
        subs_on = np.argwhere(mask)
        physical_subs.append(subs_on)

    # Assemble lambda trajectory for each frame
    lambdasPerFrame = np.concatenate(lambdasPerFrame,1)

    # Assemble array of physical ligands at each frame
    subs_on = physical_subs[0]
    if nsites != 1:
        # Get full physical end states only
        physical_subs = take_overlap(*physical_subs)
        subs_on = physical_subs[0]
        for arr in physical_subs[1:]: 
            subs_on = np.append(subs_on,np.reshape(arr[:,1],(subs_on.shape[0],1)),1)
    return subs_on, lambdasPerFrame 

def streamFileToCHARMM(FilePath):
    """
    Read a stream file in CHARMM via pyCHARMM lingo
    In:
    FilePath  (str): path to file to stream

    Out:
    CHARMM output in stdout
    """
    # Open file
    with open(FilePath,'r') as f:
        stream = f.readlines()

    # Exclude title from stream
    stream = [line for line in stream if not line.startswith('*')]
    stream = ''.join(stream)

    # Stream
    lingo.charmm_script(stream)    

def grab_frames(sysname, nsubs, subs_on, StructureFile, nsteps, nsavc, ns, eqS, ini, dup, MSLDInpPath=f'./prep/', NoSolv=False):
    """
    Given system specific properties and output from get_subs_on, return 
    a directory with frames for each physical ligand.

    In:
        sysname               (str) : MSLD system name
        nsubs        [int, int,...] : list of nsubs per site
        subs_on             (float) : np.array of size (nframes x sum(nsubs)+1)
                                      denotes frame index at col1 and subsituent
                                      index at each site on the rest of the cols
        StructureFile         (str) : path to MSLD system PSF
        nsteps                (int) : number of steps per ns trajectory
        nsavc                 (str) : freq of saving coordinates
        ns                    (int) : total number of ns
        eqS                   (int) : number of ns to discard as equilibration
        ini                   (int) : ALF iteration from which to grab trajectories 
        dup                   (str) : letter a-z denoting which duplicate to grab
                                      trajectory from 
        MSLDInpPath           (str) : path to MSLD {sysname}.inp file
        NoSolv               (bool) : option to exclude solvent from frames 
                                      (excludes water and ions)

    Out:
        subs_on      2-D np.array  : subs_on matrix

    """
    # Define total number of frames per ns
    nf = int(nsteps/nsavc)

    # Split subs_on 
    physicalFrames = subs_on[:,0]
    subs = subs_on[:,1:]
    
    # Make frames direcotry if it doesn't exist 
    framesPath = f'./run{ini}{dup}_frames' 
    if not os.path.exists(framesPath):
        os.mkdir(framesPath)

    # Define trajectory path
    TrajPath = f'./run{ini}{dup}/dcd/'

    # Stream MSLD {sysname}.inp and associated stream files 
    variablesFile = f'./variables{ini}.inp' 
    MSLDInpFile = os.path.join(MSLDInpPath,f'{sysname}.inp')

    streamFileToCHARMM(variablesFile)
    streamFileToCHARMM(MSLDInpFile)


    # Extract physical frames
    iframe = 0                 # current overall frame across all ns trajectories
    for i in range(eqS+1,ns+1):
        # Define ns trajectory file name
        TrajFile = os.path.join(TrajPath,f'{sysname}_prod{i}.dcd_{rep}')

        # Iterate over frames until we hit a physical one
        for j in range(0,nf):  # j is the current ns frame index. It is iframe % nf
            if iframe in physicalFrames:
                # Get substituent indices per site for physical frame 
                subb = subs[iframe,:]
                sub = '_'.join([str(x+1) for x in subb])

                # Create directory for physical ligand
                physSubDir = os.path.join(framesPath,sub)
                if not os.path.exists(physSubDir):
                    os.mkdir(physSubDir)
               
                # Define pdb file path to be created 
                pdbFileName = os.path.join(physSubDir,f'{iframe}.pdb')
                
                print(iframe)
                print(j)
               
                # Open traj and read up to physical frame 
                lingo.charmm_script(f'''open unit 51 read unform name {TrajFile}
                    traj first 51 nunit 1 skip {nsavc}
                    ''')
                
                for ii in range(0,j+1):
                    lingo.charmm_script('''
                               traj read
                    ''')
                
                # Create atom selection 
                dontInclude = []
                for site, sub in enumerate(nsubs):
                    dontInclude.extend([f'site{site+1}_sub{s+1}' for _,s in enumerate(range(sub)) if s != subb[site]])
                dontInclude = ' .or. -\n'.join(dontInclude)
                selection = f'sele .not. ({dontInclude}) end'
                if NoSolv:
                    selection = f'sele .not. ({dontInclude}) .and. (.not. (segid IONS .or. segid SOLV .or. segid W*T*)) end'
              
                # Write pdb file 
                lingo.charmm_script(f'coor orient')
                lingo.charmm_script(f'''open write card unit 12 name {pdbFileName}
                    write coor {selection} pdb unit 12
                    ''')
               

                # Delete all atoms, close all units and reload PSF 
                psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())
                
                lingo.charmm_script('''close unit 51
                    close unit 12
                ''')
                
                settings.set_bomb_level(-1)
                read.psf_card(StructureFile, append = True) # Bypass image centering issues
                settings.set_bomb_level(0)
                
 
            iframe += 1

if __name__ == '__main__':
    # Define variables
    sysname = 'jnk1'  # System name
    cutoff = 0.90     # Only consider lambda >= cutoff as physical
    ns = 6            # Total number of ns (including equilibration time)
    eqS = 5           # Equilibration time (first `eqS` discarded from `ns`)
    nsteps = 500000   # Number of steps per ns run
    nsavc = 10000     # Frequency of saving frames
    nsavl = 10        # Frequency of saving lambdas
    nsubs = [4]       # [nsubs1, nsubs2, ...] List of number of subs at each site
    ini= 65           # ALF run number
    dup= 'b'          # Production duplicate to process
    rep = 0           # Replica number (0 if no replica exchange) 
    StructureFile='./prep/minimized.psf'   # MSLD System PSF
    

    nf = nsteps/nsavc
    inx=string.ascii_lowercase.index(dup)  # zero-based index
    LambdaFile=f'analysis{ini}/data/Lambda.{inx}.0.dat'
    skip = int(nsavc/nsavl)                # save frequency of traj wrt save freq of lambdas

    

    # Run
    subs_on, lambdasPerFrame = get_subs_on(LambdaFile, nsubs, nsavc=nsavc,\
                                           nsavl=nsavl, cutoff=cutoff)
    grab_frames(sysname, nsubs, subs_on, StructureFile,\
                nsteps, nsavc, ns, eqS, ini, dup)

