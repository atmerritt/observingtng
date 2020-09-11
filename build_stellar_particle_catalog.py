#!/usr/bin/env python
""" build_stellar_particle_catalog.py -- Given a SubfindID at some SnapNum, track all its constituent particles back in time.

Usage:
    build_stellar_particle_catalog.py [-h] [-v] [-w] [-O] [--snapnum INT] [--littleh FLT] [--tng STR] [--TNGdir DIR] [--outdir DIR]  <SubfindInput>

Options:
    -h, --help                           Show this screen                                               [default: False]
    -v, --verbose                        Print extra information                                        [default: False]
    -w, --superverbose                   Print even more information                                    [default: False]
    -O, --overwrite                      Overwrite existing files                                       [default: False]

    -s INT, --snapnum INT                Snap number to trace back from (99 corresponds to z=0)         [default: 99]
    -l FLT, --littleh FLT                Value of "little h"                                            [default: 0.6774]
    -L STR, --tng STR                    Which TNG are we using?                                        [default: L75n1820TNG]

    -t DIR, --TNGdir DIR                 Directory where TNG output information lives                   [default: /virgo/data/IllustrisTNG/]
    -o DIR, --outdir DIR                 Directory where particle catalogs will live                    [default: /u/allim/Projects/TNG/Subhalos/particles/] 

Example:
    python build_stellar_particle_catalog.py -v -w 505333
    python build_stellar_particle_catalog.py -v -w /u/allim/Projects/TNG/Sample/lists/testme.txt

    python build_stellar_particle_catalog.py -v -w -t /virgo/data/IllustrisTNG/L75n455TNG/ 10 
    python build_stellar_particle_catalog.py -v -w -t /virgo/data/IllustrisTNG/L75n455TNG/ /u/allim/Projects/TNG/Sample/lists/testme_lowres.txt

    python build_stellar_particle_catalog.py -v -w -t /virgo/data/IllustrisTNG/L75n910TNG/ 3 (--> when debugging via medium res runs; sIDs: 3, 10, 15)

"""

from __future__ import division

from memory_profiler import profile
from astropy.io import ascii

import illustris_python as il
import pandas as pd
import numpy as np
import numexpr
import datetime
import docopt
import h5py
import sys
import os
import gc


def print_verbose_string(asverbose):
    print >> sys.stderr,'VERBOSE: %s' % asverbose

def read_file(fname):
    arr = ascii.read(fname)
    return arr

def mkdirp(dirpath):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return None

def clean_list(slist):

    if overwrite:
        slist_cleaned = np.array(slist)

    else:
        slist_cleaned = []
        for sid in slist:
            outname = '{dir}/subhalo_{sid}_particles_history.hdf5'.format(dir=outdir,sid=str(int(sid)))
            if os.path.isfile(outname):
                print '*** Output file already exists for SubfindID '+str(sid)+' and we are not in overwrite mode - skipping! ***'
            else:
                slist_cleaned.append(sid)

        slist_cleaned = np.array(slist_cleaned)


    return slist_cleaned


def exclude_wind_particles(subhalostarsandwinds):
    subhalostars = {}
    for keyname in subhalostarsandwinds.keys():
        if keyname == 'count':
            subhalostars[keyname] = len((subhalostarsandwinds['ParticleIDs'])[subhalostarsandwinds['GFM_StellarFormationTime'] > 0])
        else:
            subhalostars[keyname] = (subhalostarsandwinds[keyname])[subhalostarsandwinds['GFM_StellarFormationTime'] > 0]

    return subhalostars


def load_subhalo_information(subfindID):
    'read in subhalo catalog'
    if verbose:
        print_verbose_string('Reading in particle data for Subhalo '+str(subfindID)+'...')
    subhalostarsandwinds = il.snapshot.loadSubhalo(os.path.join(TNGdir,'output'),snapnumbase,subfindID,'stars')

    if verbose:
        print_verbose_string('Removing wind particles ...')
    subhalostars = exclude_wind_particles(subhalostarsandwinds)

    'clean up'
    del subhalostarsandwinds
    gc.collect()

    return subhalostars



def load_my_snapshot(snapnum):
    if verbose:
        print_verbose_string('Reading in ParticleIDs and Coordinates for snapshot '+str(snapnum)+'...')
    snapcat = il.snapshot.loadSubset(os.path.join(TNGdir,'output'),snapnum,ptNumStars,fields=['ParticleIDs','Coordinates'])

    'single precision'
    if snapcat['count'] > 0:
        snapcat['Coordinates'] = snapcat['Coordinates'].astype(np.float32)
        
        'sort by particle indices, but KEEP TRACK OF THE ORIGINAL SORTING'
        snapcat['original_indices'] = np.arange(snapcat['count'],dtype=int)
        snapcat['ParticleIDs_nosort'] = np.array(snapcat['ParticleIDs'])
        sorted_inds = np.argsort(snapcat['ParticleIDs'])

        snapcat['original_indices'] = snapcat['original_indices'][sorted_inds]
        snapcat['ParticleIDs'] = snapcat['ParticleIDs'][sorted_inds]
        snapcat['Coordinates'] = snapcat['Coordinates'][sorted_inds,:]

        del sorted_inds

    return snapcat



def load_my_subhalos(snapnum):
    if verbose:
        print_verbose_string('Reading in subhalo group catalogs for snapshot '+str(snapnum)+'...')
    subcat = il.groupcat.loadSubhalos(os.path.join(TNGdir,'output'),snapnum,fields=['SubhaloLenType','SubhaloMassType'])

    if subcat['count'] > 0:
        subcat['SubhaloLenStars'] = subcat['SubhaloLenType'][:,ptNumStars]
        subcat['SubhaloMassStars'] = subcat['SubhaloMassType'][:,ptNumStars]
    return subcat



def read_my_offsets(snapnum):
    if verbose:
        print_verbose_string('Reading offsets for snapshot '+str(snapnum)+'...')

    offcat = {}
    f = h5py.File('{tngdir}/postprocessing/offsets/offsets_{sn}.hdf5'.format(tngdir=TNGdir,sn=str(snapnum).zfill(3)),'r')
    if 'Subhalo' in f.keys():
        offs = f.get('Subhalo/SnapByType').value
        offcat[str(snapnum)] = offs[:,ptNumStars]
    f.close()

    return offcat


def prep_tree_catalog(SubfindIDlist):
    if verbose:
        print_verbose_string('Setting up tree structure ..')

    Nsubs = len(SubfindIDlist)

    'set up tree catalog'
    treecat = {}
    treecat['SubfindID'] = np.array(SubfindIDlist)
    for sn in range(100):
        treecat[str(sn)+'/SubhaloPos'] = -1*np.ones(3*Nsubs).reshape((Nsubs,3))
        treecat[str(sn)+'/Group_R_Crit200'] = -1*np.ones(Nsubs)
        treecat[str(sn)+'/SubfindID'] = -1*np.ones(Nsubs)
        treecat[str(sn)+'/SubhaloLenStars'] = -1*np.ones(Nsubs)
        treecat[str(sn)+'/SubhaloHalfMassRadStars'] = -1*np.ones(Nsubs)
        treecat[str(sn)+'/SubhaloMassStars'] = -1*np.ones(Nsubs)

    'fill it in'
    for ind,subfindID in enumerate(SubfindIDlist):
        'load merger tree (main branch only)'
        treeMBP = il.sublink.loadTree(os.path.join(TNGdir,'output'),snapnumbase,subfindID,treeName='SubLink_gal',onlyMPB=True)
        for snapnum in range(100):
            if snapnum in treeMBP['SnapNum']:
                treecat[str(snapnum)+'/SubhaloPos'][ind] = treeMBP['SubhaloPos'][treeMBP['SnapNum']==snapnum] # ckpc/h
                treecat[str(snapnum)+'/Group_R_Crit200'][ind] = treeMBP['Group_R_Crit200'][treeMBP['SnapNum']==snapnum] # ckpc/h
                treecat[str(snapnum)+'/SubfindID'][ind] = treeMBP['SubfindID'][treeMBP['SnapNum']==snapnum] 
                treecat[str(snapnum)+'/SubhaloLenStars'][ind] = treeMBP['SubhaloLenType'][treeMBP['SnapNum']==snapnum,ptNumStars] 
                treecat[str(snapnum)+'/SubhaloHalfMassRadStars'][ind] = treeMBP['SubhaloHalfmassRadType'][treeMBP['SnapNum']==snapnum,ptNumStars]
                treecat[str(snapnum)+'/SubhaloMassStars'][ind] = treeMBP['SubhaloMassType'][treeMBP['SnapNum']==snapnum,ptNumStars]
        del treeMBP
        
    gc.collect()
    
    return treecat


def prep_particle_catalog(SubfindIDlist):
    if verbose:
        print_verbose_string('Setting up particle catalog structure ..')

    keylist = ['ParticleIDs','SubfindIDs_99','ParticleMasses',
               'snap_form','snap_firstvircross','snap_strip',
               'dsubsq_form','dsubsq_firstvircross','dsubsq_strip','dsubsq_last',
               'mstell_form','mstell_strip','mstell_strip_prev',
               'coo_last',
               'host_form','host_strip','host_strip_prev',
               '_inds_prev_','_mstell_prev_','_host_subfindids_prev_']

    partcat = {}
    for keyname in keylist:
        if 'coo' in keyname:
            partcat[keyname] = np.empty((0,3))
        elif 'IDs' in keyname:
            partcat[keyname] = np.empty(0,dtype=int)
        else:
            partcat[keyname] = np.empty(0)

    for subfindID in SubfindIDlist:
        'read in subhalo data'
        subhalostars = load_subhalo_information(subfindID)

        'fill in default/placeholder information'
        partcat['ParticleIDs'] = np.concatenate(( partcat['ParticleIDs'], subhalostars['ParticleIDs'] ))
        partcat['ParticleMasses'] = np.concatenate(( partcat['ParticleMasses'], subhalostars['Masses'] ))
        partcat['SubfindIDs_99'] = np.concatenate(( partcat['SubfindIDs_99'], subfindID*np.ones(subhalostars['count']) ))
        partcat['coo_last'] = np.concatenate(( partcat['coo_last'], subhalostars['Coordinates'] ))

        for key in ['snap_firstvircross','snap_strip','snap_form']:
            partcat[key] = np.concatenate(( partcat[key],-999*np.ones(subhalostars['count']) ))

        for key in ['dsubsq_firstvircross','dsubsq_strip','dsubsq_form','dsubsq_last']:
            partcat[key] = np.concatenate(( partcat[key],-999*np.ones(subhalostars['count']) ))

        for key in ['mstell_strip','mstell_strip_prev','mstell_form']:
            partcat[key] = np.concatenate(( partcat[key],-999*np.ones(subhalostars['count']) ))

        for key in ['host_strip','host_strip_prev','host_form']:
            partcat[key] = np.concatenate(( partcat[key],-999*np.ones(subhalostars['count']) ))

        for key in ['_inds_prev_','_mstell_prev_','_host_subfindids_prev_']:
            partcat[key] = np.concatenate(( partcat[key],-1*np.ones(subhalostars['count']) ))

        del subhalostars

    'sort by ParticleID'
    partcat['original_indices'] = np.arange(len(partcat['ParticleIDs']),dtype=int)
    sorted_inds = np.argsort(partcat['ParticleIDs'])
    for key in partcat.keys():
        partcat[key] = partcat[key][sorted_inds]

    partcat['sortback_inds'] = np.argsort(partcat['original_indices'])

    del sorted_inds
    gc.collect()

    return partcat

def write_hdf5(partcat,treecat,outdir):
    'write out a file for each subhalo'
    subIDs = np.unique(partcat['SubfindIDs_99'])
    for ind,subfindID in enumerate(subIDs):
        if verbose:
            print_verbose_string('Writing hdf5 file for SubfindID '+str(int(subfindID))+' ('+str(ind+1)+' of '+str(len(subIDs))+')...')

        outname = '{dir}/subhalo_{sid}_particles_history_snap{snapnum}.hdf5'.format(dir=outdir,sid=str(int(subfindID)),snapnum=str(snapnumbase))
        if verbose:
            print_verbose_string('---> '+outname)

        h5out = h5py.File(outname,'w')
    
        'Add particle catalog data'
        for key in partcat.keys():
            h5out.create_dataset(key,data=(partcat[key])[partcat['SubfindIDs_99']==subfindID],compression='gzip',compression_opts=9,chunks=True)

        'Add merger tree catalog data'
        snaps = np.arange(100)
        h5out.create_dataset('tree_SnapNum',data=snaps,compression='gzip',compression_opts=9,chunks=True)
        h5out.create_dataset('tree_Group_R_Crit200',data=np.array([treecat[str(sn)+'/Group_R_Crit200'][treecat['SubfindID']==subfindID][0] for sn in snaps]),
                             compression='gzip',compression_opts=9,chunks=True)
        h5out.create_dataset('tree_SubhaloMassStars',data=np.array([treecat[str(sn)+'/SubhaloMassStars'][treecat['SubfindID']==subfindID][0] for sn in snaps]),
                             compression='gzip',compression_opts=9,chunks=True)
        h5out.create_dataset('tree_SubhaloHalfMassRadStars',data=np.array([treecat[str(sn)+'/SubhaloHalfMassRadStars'][treecat['SubfindID']==subfindID][0] for sn in snaps]),
                             compression='gzip',compression_opts=9,chunks=True)

        h5out.close()

    return None


def get_host_properties(inds_nosort,offcat,subcat,snapnum):

    'make sure you are only dealing with offset indices (and thereby subfind ids and first particle indices) where stellar particles exist!'
    firstptinds_withstars = offcat[str(snapnum)][subcat['SubhaloLenStars'] > 0]
    Noffs = len(offcat[str(snapnum)])
    subfind_ids_withstars = np.arange(Noffs)[subcat['SubhaloLenStars'] > 0]

    'figure out which bins the particle inds fall into; use that to find host IDs'
    host_offset_inds = np.digitize(inds_nosort,firstptinds_withstars)-1
    host_subfind_ids = subfind_ids_withstars[host_offset_inds]

    'now just add up the masses .. '
    host_mstell = (subcat['SubhaloMassStars'])[host_subfind_ids]

    'clean up'
    del firstptinds_withstars
    del subfind_ids_withstars
    del host_offset_inds
    gc.collect()

    return host_mstell,host_subfind_ids



@profile
def track_stellar_particles(snapnum,partcat,treecat,snapcat,subcat,offcat):

    if verbose:
        print_verbose_string('Tracking particles across Snapshot '+str(snapnum)+'...')


    '''
    (1) READ IN RELEVANT INFORMATION ABOUT THIS SNAPSHOT
    - find particle indices
    - grab particle coordinates
    - grab coordinates and Rvir for MBP
    - determine host masses
    '''

    if 'ParticleIDs' not in snapcat.keys():
        print '-> skipping! literally nothing here.'
        return None

    Nptcls = len(snapcat['ParticleIDs'])
    inds = np.in1d(partcat['ParticleIDs'],snapcat['ParticleIDs']).astype(int)
    nz = np.nonzero(np.in1d(snapcat['ParticleIDs'],partcat['ParticleIDs']))[0]
    inds[inds==0] = -1
    inds[inds==1] = nz
    inds_nosort = snapcat['original_indices'][inds]

    if len(inds[inds>=0]) == 0:
        print '-> skipping! no particles at this snapshot!'
        del inds
        return None


    partcat['snap_form'][(partcat['snap_form'] < 0) & (inds >= 0)] = snapnum

    coords = snapcat['Coordinates'][inds,:] # ckpc/h
    coords_sub = np.array([ treecat[str(snapnum)+'/SubhaloPos'][treecat['SubfindID']==sid][0] for sid in partcat['SubfindIDs_99'] ]) # ckpc/h
    Rvir_fof = np.array([ treecat[str(snapnum)+'/Group_R_Crit200'][treecat['SubfindID']==sid][0] for sid in partcat['SubfindIDs_99'] ]) # ckpc/h

    host_mstell,host_subfindids = get_host_properties(inds_nosort,offcat,subcat,snapnum)



    '''
    (2) DETERMINE WHETHER PARTICLES HAVE CROSSED THE VIRIAL RADIUS OF THE MBP
    - calculate distance to the center of mass for the FoF associated with the MBP
    - only update the first time virdist_sq <= Rvir**2!
    '''
        
    subdist_sq = numexpr.evaluate('sum((coo - coo_sub)**2,axis=1)',local_dict={"coo":coords,"coo_sub":coords_sub})
    invir = np.array(subdist_sq <= Rvir_fof**2)

    partcat['dsubsq_firstvircross'][(invir == True) & (partcat['snap_firstvircross'] < 0) & (inds >= 0)] = subdist_sq[(invir == True) & (partcat['snap_firstvircross'] < 0) & (inds >= 0)]
    partcat['snap_firstvircross'][(invir == True) & (partcat['snap_firstvircross'] < 0 ) & (inds >= 0)] = snapnum



    '''
    (3) DETERMINE WHETHER PARTICLES ARE BOUND TO THE MBP
    - identify all particles bound to MBP at this snapshot
    - determine whether particles of interest are included in that last
    '''

    'find host subfindIDs for each particle at this snapshot'
    MBP_sids = np.array([ treecat[str(snapnum)+'/SubfindID'][treecat['99/SubfindID']==sid][0] for sid in partcat['SubfindIDs_99'] ])

    'index the offset catalog with these IDs to find the first particle index'
    MBP_inds0 = np.array([ int(offcat[str(snapnum)][mbp]) for mbp in MBP_sids ])

    'how many particles in each of these subhalos?'
    MBP_Ns = np.array([ treecat[str(snapnum)+'/SubhaloLenStars'][treecat['99/SubfindID']==sid][0] for sid in partcat['SubfindIDs_99'] ])

    'bound?'
    isbound = np.array([ True if pid in snapcat['ParticleIDs_nosort'][range(int(MBP_ind0),int(MBP_ind0+MBP_N+1))] else False for pid,MBP_ind0,MBP_N in zip(partcat['ParticleIDs'],MBP_inds0,MBP_Ns) ])

    partcat['dsubsq_strip'][(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)] = subdist_sq[(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)]
    partcat['mstell_strip'][(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)] = host_mstell[(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)]
    partcat['mstell_strip_prev'][(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)] = partcat['_mstell_prev_'][(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)]
    partcat['host_strip'][(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)] = host_subfindids[(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)]
    partcat['host_strip_prev'][(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)] = partcat['_host_subfindids_prev_'][(isbound == True) & (partcat['snap_strip'] < 0) & (inds >= 0)]
    partcat['snap_strip'][(isbound == True) & (partcat['snap_strip'] < 0 ) & (inds >= 0)] = snapnum



    '''
    (4) RECORD RELEVANT INFORMATION FOR PARTICLES AT FORMATION TIME
    '''

    partcat['dsubsq_form'][(partcat['snap_form']==snapnum) & (inds >=0)] = subdist_sq[(partcat['snap_form']==snapnum) & (inds >=0)]
    partcat['mstell_form'][(partcat['snap_form']==snapnum) & (inds >=0)] = host_mstell[(partcat['snap_form']==snapnum) & (inds >=0)]
    partcat['host_form'][(partcat['snap_form']==snapnum) & (inds >=0)] = host_subfindids[(partcat['snap_form']==snapnum) & (inds >=0)]


    '''
    (4) RECORD RELEVANT INFORMATION FOR PARTICLES AT PRESENT TIME (SNAP 99)
    '''

    if snapnum == 99:
        partcat['dsubsq_last'][inds >=0] = subdist_sq[inds >=0]
            
    'keep track of indices and stellar masses from the previous snapshot'
    partcat['_inds_prev_'] = np.array(inds)
    partcat['_mstell_prev_'] = np.array(host_mstell)
    partcat['_host_subfindids_prev_'] = np.array(host_subfindids)

    'clear out everything else'
    del Nptcls
    del inds
    del inds_nosort
    del nz
    del coords
    del coords_sub
    del Rvir_fof
    del host_mstell
    del host_subfindids
    del subdist_sq
    del invir
    del MBP_sids
    del MBP_inds0
    del MBP_Ns
    del isbound
    gc.collect()

    return None


@profile
def loop_over_snapshots(partcat,treecat):
    """
    Loop over all snapshots; for each one, keep track of relevant information for each subhalo.
    """

    'set up defaults for previous indices etc ...'

    'goooo '
    for snapnum in range(snapnumbase):

        'load particle data for all snapshots'
        t0 = datetime.datetime.now()
        snapcat = load_my_snapshot(snapnum)
        t1 = datetime.datetime.now()
        print_verbose_string('Loading particle data from snapshot '+str(snapnum)+' took '+str(np.around((t1-t0).total_seconds()/60.,2))+' minutes')
        
        'check for particle data...'
        if snapcat['count'] == 0:
            print '*** No stellar particles found in Snapshot '+str(snapnum)+'! Moving on ...\n'
            del snapcat
            continue

        'load particle data for all snapshots'
        subcat = load_my_subhalos(snapnum)
        t2 = datetime.datetime.now()
        print_verbose_string('Loading subhalo group catalogs from snapshot '+str(snapnum)+' took '+str(np.around((t2-t1).total_seconds()/60.,2))+' minutes')

        'load offsets for all snapshots'
        offcat = read_my_offsets(snapnum)
        t3 = datetime.datetime.now()
        print_verbose_string('Loading offsets for snapshot '+str(snapnum)+' took '+str(np.around((t3-t2).total_seconds()/60.,2))+' minutes')

        'track particles and keep track of relevant info'
        track_stellar_particles(snapnum,partcat,treecat,snapcat,subcat,offcat)
        t4 = datetime.datetime.now()
        print_verbose_string('Tracking particles across snapshot '+str(snapnum)+' took '+str(np.around((t4-t3).total_seconds()/60.,2))+' minutes')

        'delete snapcat and subcat to clear up memory'
        del snapcat
        del subcat
        del offcat

        'clear out memory garbage stuff'
        gc.collect()

        print ''

    write_hdf5(partcat,treecat,outdir)

    return None


@profile
def main():
    'create temporary files to keep track of what is currently running'
    for subfindID in SubfindIDlist:
        open(os.path.join(outdir, 'RUNNING_subhalo_%s_particles_history_snap99.hdf5' % subfindID), 'a').close()

    'set up catalogs'
    partcat = prep_particle_catalog(SubfindIDlist)
    ta = datetime.datetime.now()
    print_verbose_string('Prepping particle catalogs for all subhalos and snapshots took '+str(np.around((ta-tstart).total_seconds()/60.,2))+' minutes')

    treecat = prep_tree_catalog(SubfindIDlist)
    tb = datetime.datetime.now()
    print_verbose_string('Prepping merger tree catalogs for all subhalos and snapshots took '+str(np.around((tb-ta).total_seconds()/60.,2))+' minutes')

    'fill in particle catalog'
    loop_over_snapshots(partcat,treecat)

    'delete temporary files'
    for subfindID in SubfindIDlist:
        if os.path.isfile(os.path.join(outdir, 'RUNNING_subhalo_%s_particles_history_snap99.hdf5' % subfindID)):
            os.remove(os.path.join(outdir, 'RUNNING_subhalo_%s_particles_history_snap99.hdf5' % subfindID))

    return None
    


if __name__ == '__main__':
    'enable automatic garbage collection'
    gc.enable()

    'ok carry on ..'
    arguments = docopt.docopt(__doc__)

    'options'
    verbose = arguments['--verbose']
    superverbose = arguments['--superverbose']
    overwrite = arguments['--overwrite']

    'directory structure'
    myTNG = arguments['--tng']
    TNGdir = os.path.join(arguments['--TNGdir'], myTNG)
    outdir = os.path.join(arguments['--outdir'], myTNG)
    mkdirp(outdir)


    'little h'
    h = np.float(arguments['--littleh'])
    
    'snap number to trace back from'
    snapnumbase = int(arguments['--snapnum'])

    'subhalo in question'
    SubfindInput = arguments['<SubfindInput>']

    if verbose:
        print '\n',arguments,'\n'

    'single subhalo or list of subhalos?'
    try:
        SubfindIDlist = [int(SubfindInput)]
    except ValueError:
        print 'We have a list of subhalos to work with!'
        SubfindIDlist = pd.read_csv(SubfindInput).subfindID.values


    'clean up subhalo list (avoid repeats, unless --overwrite specified)'
    'go in reverse order to start with the lowest mass galaxies'
    SubfindIDlist = clean_list(SubfindIDlist)[::-1]

    'various mass indices'
    ptNumStars = il.snapshot.partTypeNum('stars')
    ptNumDM = il.snapshot.partTypeNum('dm')

    'ready steady go' 
    tstart = datetime.datetime.now()

    main()

    tend = datetime.datetime.now()
    t_mins = (tend-tstart).total_seconds()/60.
    t_hrs =  (tend-tstart).total_seconds()/3600.

    print_verbose_string('\nTracking particles for '+str(len(SubfindIDlist))+' subhalo(s) took '+str(np.around(t_mins,2))+' minutes ('+str(np.around(t_hrs,2))+' hrs)')
    print '\nDone.\n'





