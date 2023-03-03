# Set of functions to analyze and view MSLD trajectories
# Written by: Luis F Cervantes and Furyal Ahmed (2/23)

from pymol import cmd
import numpy as np
import sys
import string
import os
from functools import reduce
import shutil

sysname = 'jnk1'  # System name
cutoff = 0.90     # Only consider lambda >= cutoff as physical
ns = 50           # Total number of ns
eqS = 5           # Equilibration time (first `eqS` discarded from `ns`)
nsteps = 500000   # Number of steps per ns run
nsavc = 10000     # Frequency of saving frames
nsavl = 10        # Frequency of saving lambdas
nsubs = [4]       # [nsubs1, nsubs2, ...] List of number of subs at each site
ini= 65           # ALF run number
dup= 'a'          # Production duplicate to process
rep = 0           # Replica number (0 if no replica exchange)
LambdaPath = './'
psfPath = './'
trajPath = './'
sysInpPath = './'
# LambdaPath=f'./analysis{ini}/data/'
# psfPath = './prep/'
# trajPath = f'./run{ini}{dup}/dcd/'
# sysInpPath = './prep/'

inx=string.ascii_lowercase.index(dup) # zero-based index
LambdaFile=os.path.join(LambdaPath,f'Lambda.{inx}.{rep}.dat')

psfFile=os.path.join(psfPath,'minimized.psf')
sysInpFile= os.path.join(sysInpPath, f'{sysname}.inp')

skip = int(nsavc/nsavl) # save frequency of traj wrt save freq of lambdas

def take_overlap(*input):
    """
    Given an unpacked list `input` of np.arrays,
    take the overlap of the arrays given the first
    column of the arrays.

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
    assert (nsavc/nsavl).is_integer, f"Frequency of saving lambdas and frames are not multiples of each other"
    nsites = len(nsubs)
    skip = int(nsavc/nsavl)
    lams = np.loadtxt(LambdaFile)
    physical_subs = []
    lambdasPerFrame = []
    for site in range(nsites):
        if site == 0:
            index1 = 0
        else:
            index1 = np.cumsum(nsubs[:site+1])[site-1]
    
        index2 = np.cumsum(nsubs[:site+1])[-1] -1  
     
        lams_site = lams[skip-1::skip,index1:index2+1]
        lambdasPerFrame.append(lams_site)
        mask = lams_site >= cutoff
        subs_on = np.argwhere(mask)
        physical_subs.append(subs_on)

    lambdasPerFrame = np.stack(*lambdasPerFrame, 0) 
    subs_on = physical_subs[0]
    if nsites != 1:
        physical_subs = take_overlap(*physical_subs) # Only get fully physical end states
        subs_on = physical_subs[0]
        for arr in physical_subs[1:]: 
            subs_on = np.append(subs_on,np.reshape(arr[:,1],(subs_on.shape[0],1)),1)
    return subs_on, lambdasPerFrame 

def get_selections(sysInpFile):
    with open(sysInpFile,'r') as f:
        lines = f.readlines()

    # This method takes care of commented out !define lines and indentations.
    # Do not think this will break. If BLOCK atom definitions start with
    # something different than site{}_sub{}, then beginds should be changed
    # accordingly. 
    beginds = [i for i,l in enumerate(lines) if l.startswith('define site')]
    endinds = [i for i,l in enumerate(lines) if l.startswith('   none ) end')]
    atSels = []
    groupNames = []
    for _,(i,j) in enumerate(zip(beginds,endinds)):
        site = int(lines[i].split()[1].split('site')[1].split('_')[0]) - 1
        # while len(atSels) != site + 1:
        #     atSels.append({})
        sub = int(lines[i].split()[1].split('sub')[1]) - 1
        atSel = [l.split()[3] for l in lines[i+2:j] if l.split()[0] == 'atom']
        groupNames.append(f'site{site+1}_sub{sub+1}')
        atSels.append(atSel)
    return groupNames, atSels     



def visualize_pymol(lambdasPerFrame, psfFile, trajPath, atSels, groupNames, eqS=5, nunits=1,rep=0,include_bonds=False, centerLig=True):
    """
    Only works for trajectories of the same replica. Does not display first `eqS`
    nanoseconds since there is no lambda data for them. 
    """
    # Load psf for connectivity info
    cmd.load(psfFile,'traj')
    
    # Load `nunits` ns trajectories starting at `eqS+1`
    for i in range(eqS+1,eqS+nunits+1):
        # PyMol does not like additions to dcd extension suffix. So
        # we modify and create a temporary file to load into pymol
        trajFile=os.path.join(trajPath,f'{sysname}_prod{i}.dcd_{rep}')
        loadFile = trajFile.split('.dcd')[0]+'.dcd'
        shutil.copyfile(trajFile,loadFile) 
    
        # Load ns trajectory 
        cmd.load_traj(loadFile,'traj')
    
        # Remove temp file
        os.remove(loadFile)

    # Get trajectory information
    nframes = cmd.count_states(selection='traj')
    lambdas = lambdasPerFrame[0:nframes, :]
    cmd.show('stick','resname LIG')
    if centerLig:
        cmd.intra_fit('traj and resname LIG',1)
 
    # Define objects
    for isub, atSel in enumerate(atSels):
        pymolSel = 'resname LIG and (name '
        pymolSel += ' or name '.join(atSel)
        pymolSel += ' )'
        if include_bonds:
            for iframe in range(nframes):
                lam = lambdas[iframe,isub]
                cmd.set('stick_transparency',1-lam, pymolSel, iframe+1)
        else:
            cmd.extract(groupNames[isub], pymolSel)
            # cmd.create(groupNames[isub], pymolSel)
            # cmd.remove(f'traj and {pymolSel}')
 
    set_lambdas = []
    if not include_bonds:
        for isub, groupName in enumerate(groupNames):
            for iframe in range(nframes):
                lam = lambdas[iframe,isub]
                if isub == 0:
                    set_lambdas.append(lam)
                cmd.set(f'stick_transparency', 1-lam, groupName, iframe+1)

    print(lambdas[:,0])
    print(set_lambdas)
    return None


_, lambdasPerFrame = get_subs_on(LambdaFile, nsubs, cutoff=cutoff)
groupNames, atSels = get_selections(sysInpFile)
visualize_pymol(lambdasPerFrame, psfFile, trajPath, atSels, groupNames, nunits=1, eqS=eqS)
