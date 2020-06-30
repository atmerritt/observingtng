#!/usr/bin/env python
"""select_massive_centrals.py - Create a sample of central galaxies above a given stellar mass limit.

Usage:
    select_massive_centrals [-h] [-v] [-t STR] [-s INT] [-l FLT] [-m FLT] [-d DIR] [-o DIR] 

Options:
    -h, --help                      Show this screen                                    [default: False]
    -v, --verbose                   Print extra information                             [default: False]

    -t STR, --tng STR               Which TNG?                                          [default: L75n1820TNG]
    -s INT, --snapnum INT           Which snapshot?                                     [default: 99]
    -l FLT, --littleh FLT           The value of "little h"                             [default: 0.6774]
    -m FLT, --masslim FLT           Select centrals above this mass                     [default: 1e10]


    -d DIR, --tngdir DIR            Directory where TNG postprocessing files live       [default: /virgo/data/IllustrisTNG]
    -o DIR, --outdir DIR            Directory where sample files will be saved          [default: /u/allim/Projects/TNG/Sample/lists/massive_centrals]

Example:
    python select_massive_centrals.py -v -m 1e10 -t L75n1820TNG -o /u/allim/Projects/TNG/Sample/lists/massive_centrals #TNG100
    python select_massive_centrals.py -v -m 1e10 -t L35n2160TNG -o /u/allim/Projects/TNG/Sample/lists/massive_centrals #TNG50
    python select_massive_centrals.py -v -m 1e10 -t L205n2500TNG -o /u/allim/Projects/TNG/Sample/lists/massive_centrals #TNG300

Note: 
- Stellar mass comparisons are done using the *total* subhalo stellar mass in *physical* units 

"""

from __future__ import division
import illustris_python as il

import pandas as pd
import numpy as np
import docopt
import sys
import os
import re

def print_verbose_string(asverbose):
    print >> sys.stderr,'VERBOSE: %s' % asverbose


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

    'options'
    verbose = arguments['--verbose']
    
    'directory structure'
    tngdir = arguments['--tngdir']
    outdir = arguments['--outdir']

    'simulation details/choices'
    myTNG = arguments['--tng']
    snapnum = int(arguments['--snapnum'])
    h = np.float(arguments['--littleh'])

    'lower stellar mass limit to impose'
    mstell_lower = np.float(arguments['--masslim'])
    ptNumStars = il.snapshot.partTypeNum('stars')

    if verbose:
        print '\nRunning with parameters:\n',arguments,'\n'

    if verbose:
        print_verbose_string('Loading in FoF data for %s at snapshot %s ...' % (myTNG, snapnum))
    fof = il.groupcat.loadHalos(os.path.join(tngdir,myTNG,'output'),snapnum,fields=['GroupFirstSub', 'GroupNsubs'])

    if verbose:
        print_verbose_string('Loading in subhalo data for %s at snapshot %s ...' % (myTNG, snapnum))
    subhalos = il.groupcat.loadSubhalos(os.path.join(tngdir,myTNG,'output'),snapnum,fields=['SubhaloMassType', 'SubhaloFlag'])
    n_subhalos = subhalos['count']

    if verbose:
        print_verbose_string('Identifying central subhalos for %s at snapshot %s ...' % (myTNG, snapnum))
    central_ids = fof['GroupFirstSub']

    if verbose:
        print_verbose_string('Applying lower stellar mass limit for %s at snapshot %s ...' % (myTNG, snapnum))
    mstell_phys = subhalos['SubhaloMassType'][:,ptNumStars] * (1e10/h)
    massive_central_IDs = np.array([sid for sid in central_ids if mstell_phys[sid] >= mstell_lower and subhalos['SubhaloFlag'][sid] != 0 and sid >= 0])
    massive_central_masses = np.array([mstell_phys[sid] for sid in central_ids if mstell_phys[sid] >= mstell_lower and subhalos['SubhaloFlag'][sid] != 0 and sid >= 0])

    print 'Found %s massive central galaxies for %s at snapshot %s!' % (len(massive_central_IDs), myTNG, snapnum)
    print 'Writing to output file ...'

    sample = pd.DataFrame(dict(subfindID=massive_central_IDs, mstell_phys_tot=massive_central_masses))
    sample.to_csv(os.path.join(outdir, 'sample_%s_centrals_logMstell_gt_%s_snap%s.txt' % (myTNG,np.around(np.log10(mstell_lower),2),snapnum)), index=False)
    print 'Done.\n'
