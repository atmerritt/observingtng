#!/usr/bin/env python
"""adjust_particles.py -- Given a subfindID and particle tracking results, adjust particles to explore various scenarios. Output can be fed into adaptivebox.

Usage:
    adjust_particles.py [-h] [-v] [-l FLT] [-s INT] [-d DIR] [-o DIR] [-p DIR] [-t DIR] <subfindID>

Options:
    -h, --help                               Show this screen                                         [default: False]
    -v, --verbose                            Show this screen                                         [default: False]

    -l FLT, --littleh FLT                    Value of "little h"                                      [default: 0.6774]
    -s INT, --snapnum INT                    Snap Number (99 corresponds to z=0)                      [default: 99]                                                                     

    -d DIR, --plotsdir DIR                   Directory where diagnostic plots will live               [default: /u/allim/Projects/TNG/Plots/particles/L75n1820TNG/]
    -o DIR, --outdir DIR                     Directory where updated subhalo files will live          [default: /u/allim/Projects/TNG/Subhalos/adjustments/L75n1820TNG/]
    -p DIR, --ptcldir DIR                    Directory where particle tracking results live           [default: /u/allim/Projects/TNG/Subhalos/particles/L75n1820TNG/]
    -t DIR, --TNGdir DIR                     Directory where TNG outputs live                         [default: /virgo/data/IllustrisTNG/L75n1820TNG/]

Example:
    python adjust_particles.py -v 483900
"""

from __future__ import division

import illustris_python as il
import tng_tools as tt

import matplotlib as ml
ml.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc,colorbar,lines
import matplotlib.patches as patches
import pylab

from astropy.io import ascii

import numpy as np
import datetime
import seaborn
import random
import docopt
import pylab
import h5py
import sys
import os
import re


font={'family':'serif','size':11}
rc('font',**font)

def print_verbose_string(asverbose):
    print >> sys.stderr,'VERBOSE: %s' % asverbose

def mkdirp(dirpath):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return None

def clearit(fname):
    if os.path.isfile(fname):
        os.remove(fname)
    return None


def check_TNG():
    if 'TNG' not in myTNG:
        print '\n**** Check format of input TNGdir ****\n'
        sys.exit()
    return None

def check_for_stars(subhalostarsandwinds):
    if 'ParticleIDs' not in subhalostarsandwinds.keys():
        print '\n*** Something went wrong - no star particles found for this subhalo. ***\n'
        sys.exit()
    return None

def grab_scalefactor(snapnum):
    if not os.path.isfile('snapnumbers.txt'):
        print '\n**** Where is snapnumbers.txt?? ****\n'
        sys.exit()

    conver = ascii.read('snapnumbers.txt')
    a = np.float(conver['Scalefactor'][conver['Snap']==snapnum])
    return a

def read_particle_tracker_outputs(subfindID):
    f = h5py.File(ptcldir+'/subhalo_'+str(subfindID)+'_particles_history.hdf5','r')

    ptclcat = {}
    ptclcat['particleIDs'] = f.get('ParticleIDs').value
    ptclcat['particleMasses'] = f.get('ParticleMasses').value
    
    ptclcat['snap_form'] = f.get('snap_form').value
    ptclcat['snap_cross'] = f.get('snap_firstvircross').value
    ptclcat['snap_strip'] = f.get('snap_strip').value

    ptclcat['a_form'] = np.array([map_snap2a[s] for s in ptclcat['snap_form']])
    ptclcat['a_last'] = np.ones(len(ptclcat['particleIDs']))

    ptclcat['dsub_form'] = np.sqrt(f.get('dsubsq_form').value)
    ptclcat['dsub_cross'] = np.sqrt(f.get('dsubsq_firstvircross').value)
    ptclcat['dsub_strip'] = np.sqrt(f.get('dsubsq_strip').value)
    ptclcat['dsub_last'] = np.sqrt(f.get('dsubsq_last').value)

    ptclcat['coo_last'] = f.get('coo_last').value

    ptclcat['host_strip'] = f.get('host_strip').value
    ptclcat['host_strip_prev'] = f.get('host_strip_prev').value

    ptclcat['log_mstell_strip'] = np.log10((f.get('mstell_strip').value)*((1e10)/h))
    ptclcat['log_mstell_strip_prev'] = np.log10((f.get('mstell_strip_prev').value)*((1e10)/h))

    'info from merger tree'
    tree_snaps = f.get('tree_SnapNum').value
    tree_SubhaloHalfMassRadStars = f.get('tree_SubhaloHalfMassRadStars').value # ckpc/h                                                                                                                   
    ptclcat['Re_99'] = np.float(tree_SubhaloHalfMassRadStars[tree_snaps == 99])/h

    f.close()
    
    return ptclcat

def calc_center_of_mass(subhalostars,cookey='Coordinates'):
    """                                                                                                                                                                                                      
    Units:                                                                                                                                                                                                       -- masses in 1e10Msun/h                                                                                                                                                                                      """

    x = np.sum((subhalostars[cookey][:,0])*(subhalostars['Masses']))/np.sum(subhalostars['Masses'])
    y = np.sum((subhalostars[cookey][:,1])*(subhalostars['Masses']))/np.sum(subhalostars['Masses'])
    z = np.sum((subhalostars[cookey][:,2])*(subhalostars['Masses']))/np.sum(subhalostars['Masses'])

    return x,y,z

def find_exsitu_postinfall(outcat,ptclcat):
    '''
    Just flag all exsitu particles in this subhalo that formed post-infall
    '''

    outcat['is_exsitu_postinfall'] = np.zeros(len(outcat['particleIDs']))
    outcat['is_exsitu_postinfall'][(outcat['is_insitu']==0) & (ptclcat['snap_form'] >= ptclcat['snap_cross'])] = 1.

    return outcat


def find_exsitu_massivemergers_uniform(outcat,ptclcat):
    '''
    Downsample exsitu particles that came from mergers in a certain mass range over a certain redshift range
    -- log(mprog) =< 9
    -- 2.0 >= z_acc 
    -- by a factor of 2
    '''

    outcat['possible_downsample-1'] = np.zeros(len(outcat['particleIDs']))
    #outcat['possible_downsample-1'][ (outcat['is_insitu'] == 0) & (ptclcat['snap_strip'] >= map_z2snap[2]) & (ptclcat['snap_strip'] <= map_z2snap[0.5])
    #                                     & (ptclcat['log_mstell_strip_prev'] >= 8.0) & (ptclcat['log_mstell_strip_prev'] <= 9.0) ] = 1

    outcat['possible_downsample-1'][ (outcat['is_insitu'] == 0) & (ptclcat['snap_strip'] >= map_z2snap[2]) & (ptclcat['log_mstell_strip_prev'] <= 9.0) ] = 1

    posdown_pids = outcat['particleIDs'][outcat['possible_downsample-1']==1]
    downsamp_pids = random.sample(posdown_pids, int(len(posdown_pids)/2))

    outcat['toomany_mergers-1'] = np.zeros(len(outcat['particleIDs']))
    for pID in downsamp_pids:
        outcat['toomany_mergers-1'][outcat['particleIDs']==pID] = 1.


    return outcat


def find_exsitu_massivemergers_remove_z2(outcat,ptclcat):
    '''
    mergers-5
    (modified Lars's suggestion)
    Completely remove exsitu particles that came from mergers with <= 1e9Msun, since z=2
    '''

    outcat['toomany_mergers-5'] = np.zeros(len(outcat['particleIDs']))
    outcat['toomany_mergers-5'][ (outcat['is_insitu'] == 0) & (ptclcat['snap_strip'] >= map_z2snap[2]) & (ptclcat['log_mstell_strip_prev'] <= 9.0) ] = 1

    print 'number of particles acquired from things <= 1e9 since z=2:',len(outcat['toomany_mergers-5'][outcat['toomany_mergers-5']==1.])

    return outcat


def find_exsitu_massivemergers_remove_all(outcat,ptclcat):
    '''
    mergers-6
    (Lars's suggestion)
    Completely remove exsitu particles that came from mergers with <= 1e9Msun
    '''

    outcat['toomany_mergers-6'] = np.zeros(len(outcat['particleIDs']))
    outcat['toomany_mergers-6'][ (outcat['is_insitu'] == 0) & (ptclcat['log_mstell_strip_prev'] <= 9.0) ] = 1

    print 'number of particles acquired from things <= 1e9:',len(outcat['toomany_mergers-6'][outcat['toomany_mergers-6']==1.])

    return outcat



def find_exsitu_downsample_individual(outcat,ptclcat):
    '''
    mergers-7
    attempting to be a bit more realistic here
    find the 90th percentile of progenitor mstells in the outskirts, and randomly remove 50 percent of the particles that came from stuff below that.
    no timing constraints here?
    '''

    max_mstell = np.percentile(ptclcat['log_mstell_strip_prev'][(outcat['is_insitu'] == 0) & (ptclcat['dsub_last']/h >= 20.)], 90)

    outcat['possible_downsample-7'] = np.zeros(len(outcat['particleIDs']))
    outcat['possible_downsample-7'][ (outcat['is_insitu'] == 0) & (ptclcat['snap_strip'] >= map_z2snap[2]) & (ptclcat['log_mstell_strip_prev'] <= max_mstell)  ] = 1.

    print 'number of exsitu particles with z<2 and m<1e9:',len(outcat['possible_downsample-7'][ (outcat['is_insitu'] == 0) & 
                                                                                                (ptclcat['snap_strip'] >= map_z2snap[2]) & (ptclcat['log_mstell_strip_prev']<=9.0)  ])
    print 'number of exsitu particles with z<2 and max_mstell:',len(outcat['possible_downsample-7'][ (outcat['is_insitu'] == 0) & 
                                                                                                     (ptclcat['snap_strip'] >= map_z2snap[2]) & (ptclcat['log_mstell_strip_prev'] <= max_mstell)  ])


    posdown_pids = outcat['particleIDs'][outcat['possible_downsample-7']==1]
    downsamp_pids = random.sample(posdown_pids, int(len(posdown_pids)/2))

    outcat['toomany_mergers-7'] = np.zeros(len(outcat['particleIDs']))
    for pID in downsamp_pids:
        outcat['toomany_mergers-7'][outcat['particleIDs']==pID] = 1.
    return outcat

def find_exsitu_massivemergers_z1(outcat,ptclcat):
    '''
    Remove all exsitu particles that came from mergers more recently than z=1
    '''


    outcat['toomany_mergers-2'] = np.zeros(len(outcat['particleIDs']))
    outcat['toomany_mergers-2'][ (outcat['is_insitu']==0) & (ptclcat['snap_strip'] > map_z2snap[1]) ] = 1.

    return outcat





def find_exsitu_massivemergers_mostmassive(outcat,ptclcat):
    '''
    Identify the most massive merger from the merger trees and remove it
    '''

    'read in the tree'
    tree = il.sublink.loadTree(TNGdir+'/output',99,subfindID,treeName='SubLink_gal',onlyMPB=False)

    'find the most massive merger! need to loop through snapshots and make sure you do not include the growth of the MPB.'
    sids,masses = [],[]
    for snapnum in range(99)[::-1]:
        sids.extend(tree['SubfindID'][tree['SnapNum']==snapnum][1:])
        masses.extend(tree['SubhaloMassType'][tree['SnapNum']==snapnum,4][1:])

    sids = np.array(sids)
    masses = np.array(masses)*(1e10)/0.6774
    sorted_inds = np.argsort(masses)
    sorted_masses = masses[sorted_inds]
    sorted_sids = sids[sorted_inds]

    outcat['toomany_mergers-3'] = np.zeros(len(outcat['particleIDs']))
    use_flag = 1.
    #for mstell in sorted_masses[-3:]:
    for mstell in sorted_masses[-1:]:
        search_sid = int(sids[masses==mstell])
        snapnum_options = tree['SnapNum'][tree['SubfindID']==search_sid]
        mstell_options = tree['SubhaloMassType'][tree['SubfindID']==search_sid,4]
        search_snapnum = int(snapnum_options[mstell_options==np.max(mstell_options)])

        'load massive subhalo'
        print 'loading subhalo '+str(search_sid)+' ('+str(mstell)+')...'
        massivesubhalo = il.snapshot.loadSubhalo(TNGdir+'/output/',search_snapnum,search_sid,'stars')

        Nptcl = len(massivesubhalo['ParticleIDs'])
        Nhere = 0
        'flag particles'
        for pid in massivesubhalo['ParticleIDs']:
            outcat['toomany_mergers-3'][outcat['particleIDs']==pid] = use_flag
            if pid in outcat['particleIDs']:
                Nhere += 1

        use_flag += 1

        print '.. Nptcl:',Nptcl
        print '.. Nhere:',Nhere

    return outcat

def find_insitu(outcat,ptclcat):
    '''
    Just flag all insitu particles in this subhalo
    '''

    'insitu or nah?'
    outcat['is_insitu'] = np.array([1. if ((sf == sc) & (sf == ss)) else 0. for sf,sc,ss in zip(ptclcat['snap_form'],ptclcat['snap_cross'],ptclcat['snap_strip'])])

    return outcat

def fill_coords(outcat,subhalocat):
    ''' Copy the Coordinates from subhalocat to outcat, but make sure the order is correct'''

    outcat['CoordsOrig'] = np.zeros((outcat['particleIDs'].shape[0],3))

    for pID in outcat['particleIDs']:
        outcat['CoordsOrig'][outcat['particleIDs']==int(pID),:] = subhalocat['Coordinates'][subhalocat['ParticleIDs']==int(pID),:]

    return outcat


def contract_insitu(outcat,ptclcat,subhalocat,freduce=0.6):
    '''
    Find in-situ particles and shrink their galacto-centeric distances
    '''

    outcat['CoordsContractInSitu'] = np.array(outcat['CoordsOrig'])

    pids = outcat['particleIDs'][outcat['is_insitu']==1]
    xc,yc,zc = calc_center_of_mass(subhalocat)
    
    for ind,pID in enumerate(pids):
        'move it in!'
        x_orig = outcat['CoordsOrig'][outcat['particleIDs']==pID][0][0]
        delx = x_orig - xc
        new_x = freduce*delx + xc

        y_orig = outcat['CoordsOrig'][outcat['particleIDs']==pID][0][1] 
        dely = y_orig - yc
        new_y = freduce*dely + yc

        z_orig = outcat['CoordsOrig'][outcat['particleIDs']==pID][0][2]
        delz = z_orig - zc
        new_z = freduce*delz + zc

        dist_orig = np.sqrt((x_orig - xc)**2 + (y_orig - yc)**2 + (z_orig - zc)**2)
        dist_new = np.sqrt((new_x - xc)**2 + (new_y - yc)**2 + (new_z - zc)**2)

        outcat['CoordsContractInSitu'][outcat['particleIDs']==pID] = np.array([new_x,new_y,new_z])
    return outcat


def find_exsitu_mergers_redistribute(outcat,ptclcat,subhalocat,z=20,freduce=0.5):
    '''
    Find ex-situ particles aquired since some redshift and move them further in.
    '''

    outcat['CoordsRedistributedMergers_z'+str(z)+'_f'+str(freduce)] = np.array(outcat['CoordsOrig'])
    outcat['DistRedistributedMergers_z'+str(z)+'_f'+str(freduce)+'_sq'] = np.array(outcat['DistOrig_sq'])

    pids = outcat['particleIDs'][(outcat['is_insitu'] == 0) & (ptclcat['snap_strip'] >= map_z2snap[z])]
    xc,yc,zc = calc_center_of_mass(subhalocat)
    max_attempts = 1000

    for ind,pID in enumerate(pids):
        'sample it back to 10% of its current distance'
        d_now = np.float((ptclcat['dsub_last']*ptclcat['a_last']/h)[ptclcat['particleIDs']==pID])
        d_redistributed = freduce*d_now

        xpos = np.random.uniform( low=xc - 1./np.sqrt(3)*d_redistributed, high=xc + 1./np.sqrt(3)*d_redistributed )
        delx = (xpos-xc)/h

        n_tries = 0
        while True:
            ypos = np.random.uniform( low=yc - 1./np.sqrt(3)*d_redistributed, high=yc + 1./np.sqrt(3)*d_redistributed )
            dely = (ypos-yc)/h
            n_tries += 1
            if delx**2 + dely**2 < 0.8*d_redistributed**2 or n_tries > max_attempts:
                break

        delzmin = np.sqrt( (0.9*d_redistributed)**2 - delx**2 - dely**2 )
        delzmax = np.sqrt( (1.1*d_redistributed)**2 - (xpos - xc)**2 - (ypos - yc)**2 )
        zpos = np.random.uniform(low=delzmin,high=delzmax)*h + zc

        'make sure the CoordsDemigrated array is formatted exactly like Coords'
        outcat['CoordsRedistributedMergers_z'+str(z)+'_f'+str(freduce)][outcat['particleIDs']==pID] = np.array([xpos,ypos,zpos])
        outcat['DistRedistributedMergers_z'+str(z)+'_f'+str(freduce)+'_sq'][outcat['particleIDs']==pID] = delx**2 + dely**2 + ((zpos-zc)/h)**2

    return outcat

def find_insitu_demigrate(outcat,ptclcat,subhalocat):
    '''
    Flag insitu particles that have migrated out beyond 20 kpc (from anywhere within it); then "de-migrate" them.
    '''

    #migrated_ptcls = outcat['particleIDs'][(outcat['is_insitu'] == 1) & (ptclcat['dsub_last']*ptclcat['a_last']/h > 5*ptclcat['Re_99']) & (ptclcat['dsub_form']*ptclcat['a_form']/h < 5*ptclcat['Re_99'])]
    migrated_ptcls = outcat['particleIDs'][(outcat['is_insitu'] == 1) & (ptclcat['dsub_last']*ptclcat['a_last']/h > 2*ptclcat['Re_99']) & (ptclcat['dsub_form']*ptclcat['a_form']/h < 2*ptclcat['Re_99'])]
    migrated_ptcls = outcat['particleIDs'][(outcat['is_insitu'] == 1) & (ptclcat['dsub_last']*ptclcat['a_last']/h > 2*ptclcat['Re_99']) & (ptclcat['dsub_form']*ptclcat['a_form']/h < 20.)]


    #migrated_ptcls = outcat['particleIDs'][outcat['is_insitu'] == 1]
    outcat['CoordsDemigrated'] = np.array(outcat['CoordsOrig']) 

    xc,yc,zc = calc_center_of_mass(subhalocat)
    x_min,y_min = np.min(outcat['CoordsOrig'][:,0]),np.min(outcat['CoordsOrig'][:,1])
    x_max,y_max = np.min(outcat['CoordsOrig'][:,0]),np.min(outcat['CoordsOrig'][:,1])

    outcat['DistOrig_sq'] = (outcat['CoordsOrig'][:,0] - xc)**2 + (outcat['CoordsOrig'][:,1] - yc)**2 + (outcat['CoordsOrig'][:,2] - zc)**2
    outcat['DistDemigrated_sq'] = np.array(outcat['DistOrig_sq'])

    max_attempts = 1000

    for pID in migrated_ptcls:
        'sample it back to within 10% of its formation distance'
        ' -- choose x and y coords randomly; use these to sample (in a constrained way, where d is correct) the z coord'
        ' -- z**2 = d**2 - x**2 - y**2'
        ' -- note galaxy rotation is a thing so it is okay if we do not use the exact formation coordinates!'


        dform = np.float((ptclcat['dsub_form']*ptclcat['a_form']/h)[ptclcat['particleIDs']==pID])
        dnow = np.float((ptclcat['dsub_last']*ptclcat['a_last']/h)[ptclcat['particleIDs']==pID])

        xpos = np.random.uniform( low=xc - 1./np.sqrt(3)*dform, high=xc + 1./np.sqrt(3)*dform )
        delx = (xpos-xc)/h

        n_tries = 0
        while True:
            ypos = np.random.uniform( low=yc - 1./np.sqrt(3)*dform, high=yc + 1./np.sqrt(3)*dform )
            dely = (ypos-yc)/h
            if delx**2 + dely**2 < 0.8*dform**2 or n_tries > max_attempts:
                break
            n_tries += 1

        delzmin = np.sqrt( (0.9*dform)**2 - delx**2 - dely**2 )
        delzmax = np.sqrt( (1.1*dform)**2 - (xpos - xc)**2 - (ypos - yc)**2 )
        zpos = np.random.uniform(low=delzmin,high=delzmax)*h + zc

        'make sure the CoordsDemigrated array is formatted exactly like Coords'
        outcat['CoordsDemigrated'][outcat['particleIDs']==pID] = np.array([xpos,ypos,zpos])
        outcat['DistDemigrated_sq'][outcat['particleIDs']==pID] = delx**2 + dely**2 + ((zpos-zc)/h)**2

    return outcat


def plot_adjusted_particles(outcat,subhalocat,z=1):
    w = 0.4
    fs = 11

    fig_insitu = plt.figure(figsize=(12,6))

    axleft_insitu = fig_insitu.add_axes([0.05,0.12,w,w*(12./6)])
    axleft_insitu.set_title('All stellar particles',fontsize=fs)
    axleft_insitu.tick_params(axis='both',labelsize=fs)
    axleft_insitu.xaxis.set_ticklabels(axleft_insitu.get_xticklabels(),visible=False)
    axleft_insitu.yaxis.set_ticklabels(axleft_insitu.get_yticklabels(),visible=False)
    axleft_insitu.scatter( outcat['CoordsOrig'][:,0], outcat['CoordsOrig'][:,1], s=12, c='k' )

    axright_insitu = fig_insitu.add_axes([0.55,0.12,w,w*(12./6)])
    axright_insitu.set_title('Ex-situ particles only',fontsize=fs)
    axright_insitu.tick_params(axis='both',labelsize=fs)
    axright_insitu.xaxis.set_ticklabels(axright_insitu.get_xticklabels(),visible=False)
    axright_insitu.yaxis.set_ticklabels(axright_insitu.get_yticklabels(),visible=False)
    axright_insitu.set_xlim(axright_insitu.get_xlim())
    axright_insitu.set_ylim(axright_insitu.get_ylim())
    axright_insitu.scatter( (outcat['CoordsOrig'][:,0])[outcat['is_insitu']==0], (outcat['CoordsOrig'][:,1])[outcat['is_insitu']==0], s=12, c='k' )

    fig_insitu.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_insitu.pdf')


    print 'Number of particles remaining after getting rid of insitu particles:',len(outcat['particleIDs'][outcat['is_insitu']==0])


    fig_postinfall = plt.figure(figsize=(12,6))

    axleft_postinfall = fig_postinfall.add_axes([0.05,0.12,w,w*(12./6)])
    axleft_postinfall.set_title('All stellar particles',fontsize=fs)
    axleft_postinfall.tick_params(axis='both',labelsize=fs)
    axleft_postinfall.xaxis.set_ticklabels(axleft_postinfall.get_xticklabels(),visible=False)
    axleft_postinfall.yaxis.set_ticklabels(axleft_postinfall.get_yticklabels(),visible=False)
    axleft_postinfall.scatter( outcat['CoordsOrig'][:,0], outcat['CoordsOrig'][:,1], s=12, c='k' )

    axright_postinfall = fig_postinfall.add_axes([0.55,0.12,w,w*(12./6)])
    axright_postinfall.set_title('Excluding ex-situ,post-infall particles',fontsize=fs)
    axright_postinfall.tick_params(axis='both',labelsize=fs)
    axright_postinfall.xaxis.set_ticklabels(axright_postinfall.get_xticklabels(),visible=False)
    axright_postinfall.yaxis.set_ticklabels(axright_postinfall.get_yticklabels(),visible=False)
    axright_postinfall.set_xlim(axright_postinfall.get_xlim())
    axright_postinfall.set_ylim(axright_postinfall.get_ylim())
    axright_postinfall.scatter( (outcat['CoordsOrig'][:,0])[outcat['is_exsitu_postinfall']==0], (outcat['CoordsOrig'][:,1])[outcat['is_exsitu_postinfall']==0], s=12, c='k' )

    fig_postinfall.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_postinfall.pdf')

    print 'Number of particles remaining after getting rid of post-infall exsitu particles:',len(outcat['particleIDs'][outcat['is_exsitu_postinfall']==0])



    fig_demigrate = plt.figure(figsize=(12,12))

    axtopleft_dem = fig_demigrate.add_axes([0.05,0.55,w,w])
    axtopleft_dem.set_title('All in-situ stellar particles',fontsize=fs)
    axtopleft_dem.tick_params(axis='both',labelsize=fs)
    axtopleft_dem.xaxis.set_ticklabels(axtopleft_dem.get_xticklabels(),visible=False)
    axtopleft_dem.yaxis.set_ticklabels(axtopleft_dem.get_yticklabels(),visible=False)
    axtopleft_dem.scatter( (outcat['CoordsOrig'][:,0])[outcat['is_insitu']==1], (outcat['CoordsOrig'][:,1])[outcat['is_insitu']==1], s=12, c='k' )

    axtopright_dem = fig_demigrate.add_axes([0.55,0.55,w,w])
    axtopright_dem.set_title('De-migrated',fontsize=fs)
    axtopright_dem.tick_params(axis='both',labelsize=fs)
    axtopright_dem.xaxis.set_ticklabels(axtopright_dem.get_xticklabels(),visible=False)
    axtopright_dem.yaxis.set_ticklabels(axtopright_dem.get_yticklabels(),visible=False)
    axtopright_dem.set_xlim(axtopleft_dem.get_xlim())
    axtopright_dem.set_ylim(axtopleft_dem.get_ylim())
    axtopright_dem.scatter( (outcat['CoordsDemigrated'][:,0])[outcat['is_insitu']==1], (outcat['CoordsDemigrated'][:,1])[outcat['is_insitu']==1], s=12, c='k' )

    axbotleft_dem = fig_demigrate.add_axes([0.05,0.05,w,w])
    axbotleft_dem.set_title('All stellar particles',fontsize=fs)
    axbotleft_dem.tick_params(axis='both',labelsize=fs)
    axbotleft_dem.xaxis.set_ticklabels(axbotleft_dem.get_xticklabels(),visible=False)
    axbotleft_dem.yaxis.set_ticklabels(axbotleft_dem.get_yticklabels(),visible=False)
    axbotleft_dem.scatter( outcat['CoordsOrig'][:,0], outcat['CoordsOrig'][:,1], s=12, c='k' )

    axbotright_dem = fig_demigrate.add_axes([0.55,0.05,w,w])
    axbotright_dem.set_title('De-migrated',fontsize=fs)
    axbotright_dem.tick_params(axis='both',labelsize=fs)
    axbotright_dem.xaxis.set_ticklabels(axbotright_dem.get_xticklabels(),visible=False)
    axbotright_dem.yaxis.set_ticklabels(axbotright_dem.get_yticklabels(),visible=False)
    axbotright_dem.set_xlim(axbotleft_dem.get_xlim())
    axbotright_dem.set_ylim(axbotleft_dem.get_ylim())
    axbotright_dem.scatter( outcat['CoordsDemigrated'][:,0], outcat['CoordsDemigrated'][:,1], s=12, c='k' )

    fig_demigrate.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_demigrated.pdf')

    print 'Number of particles remaining after de-migrating insitu particles back within 5Re:',len(outcat['particleIDs'])




    fig_mergers1 = plt.figure(figsize=(12,12))

    axtopleft_mer1 = fig_mergers1.add_axes([0.05,0.55,w,w])
    axtopleft_mer1.set_title('All ex-situ stellar particles',fontsize=fs)
    axtopleft_mer1.tick_params(axis='both',labelsize=fs)
    axtopleft_mer1.xaxis.set_ticklabels(axtopleft_mer1.get_xticklabels(),visible=False)
    axtopleft_mer1.yaxis.set_ticklabels(axtopleft_mer1.get_yticklabels(),visible=False)
    axtopleft_mer1.scatter( (outcat['CoordsOrig'][:,0])[outcat['is_insitu']==0], (outcat['CoordsOrig'][:,1])[outcat['is_insitu']==0], s=12, c='k' )

    axtopright_mer1 = fig_mergers1.add_axes([0.55,0.55,w,w])
    axtopright_mer1.set_title('Downsampled mergers',fontsize=fs)
    axtopright_mer1.tick_params(axis='both',labelsize=fs)
    axtopright_mer1.xaxis.set_ticklabels(axtopright_mer1.get_xticklabels(),visible=False)
    axtopright_mer1.yaxis.set_ticklabels(axtopright_mer1.get_yticklabels(),visible=False)
    axtopright_mer1.set_xlim(axtopleft_mer1.get_xlim())
    axtopright_mer1.set_ylim(axtopleft_mer1.get_ylim())
    axtopright_mer1.scatter( (outcat['CoordsOrig'][:,0])[(outcat['toomany_mergers-1']==0) & (outcat['is_insitu']==0)], 
                            (outcat['CoordsOrig'][:,1])[(outcat['toomany_mergers-1']==0) & (outcat['is_insitu']==0)], s=12, c='k' )

    axbotleft_mer1 = fig_mergers1.add_axes([0.05,0.05,w,w])
    axbotleft_mer1.set_title('All stellar particles',fontsize=fs)
    axbotleft_mer1.tick_params(axis='both',labelsize=fs)
    axbotleft_mer1.xaxis.set_ticklabels(axbotleft_mer1.get_xticklabels(),visible=False)
    axbotleft_mer1.yaxis.set_ticklabels(axbotleft_mer1.get_yticklabels(),visible=False)
    axbotleft_mer1.scatter( outcat['CoordsOrig'][:,0], outcat['CoordsOrig'][:,1], s=12, c='k' )

    axbotright_mer1 = fig_mergers1.add_axes([0.55,0.05,w,w])
    axbotright_mer1.set_title('Downsampled mergers',fontsize=fs)
    axbotright_mer1.tick_params(axis='both',labelsize=fs)
    axbotright_mer1.xaxis.set_ticklabels(axbotright_mer1.get_xticklabels(),visible=False)
    axbotright_mer1.yaxis.set_ticklabels(axbotright_mer1.get_yticklabels(),visible=False)
    axbotright_mer1.set_xlim(axbotleft_mer1.get_xlim())
    axbotright_mer1.set_ylim(axbotleft_mer1.get_ylim())
    axbotright_mer1.scatter( (outcat['CoordsOrig'][:,0])[outcat['toomany_mergers-1']==0], (outcat['CoordsOrig'][:,1])[outcat['toomany_mergers-1']==0], s=12, c='k' )

    fig_mergers1.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers-1.pdf')

    print 'Number of partciles remaining after downsampling mergers(1):',len(outcat['particleIDs'][outcat['toomany_mergers-1']==0])



    fig_mergers2 = plt.figure(figsize=(12,12))

    axtopleft_mer2 = fig_mergers2.add_axes([0.05,0.55,w,w])
    axtopleft_mer2.set_title('All ex-situ stellar particles',fontsize=fs)
    axtopleft_mer2.tick_params(axis='both',labelsize=fs)
    axtopleft_mer2.xaxis.set_ticklabels(axtopleft_mer2.get_xticklabels(),visible=False)
    axtopleft_mer2.yaxis.set_ticklabels(axtopleft_mer2.get_yticklabels(),visible=False)
    axtopleft_mer2.scatter( (outcat['CoordsOrig'][:,0])[outcat['is_insitu']==0], (outcat['CoordsOrig'][:,1])[outcat['is_insitu']==0], s=12, c='k' )

    axtopright_mer2 = fig_mergers2.add_axes([0.55,0.55,w,w])
    axtopright_mer2.set_title('Downsampled mergers',fontsize=fs)
    axtopright_mer2.tick_params(axis='both',labelsize=fs)
    axtopright_mer2.xaxis.set_ticklabels(axtopright_mer2.get_xticklabels(),visible=False)
    axtopright_mer2.yaxis.set_ticklabels(axtopright_mer2.get_yticklabels(),visible=False)
    axtopright_mer2.set_xlim(axtopleft_mer2.get_xlim())
    axtopright_mer2.set_ylim(axtopleft_mer2.get_ylim())
    axtopright_mer2.scatter( (outcat['CoordsOrig'][:,0])[(outcat['toomany_mergers-2']==0) & (outcat['is_insitu']==0)], 
                            (outcat['CoordsOrig'][:,1])[(outcat['toomany_mergers-2']==0) & (outcat['is_insitu']==0)], s=12, c='k' )

    axbotleft_mer2 = fig_mergers2.add_axes([0.05,0.05,w,w])
    axbotleft_mer2.set_title('All stellar particles',fontsize=fs)
    axbotleft_mer2.tick_params(axis='both',labelsize=fs)
    axbotleft_mer2.xaxis.set_ticklabels(axbotleft_mer2.get_xticklabels(),visible=False)
    axbotleft_mer2.yaxis.set_ticklabels(axbotleft_mer2.get_yticklabels(),visible=False)
    axbotleft_mer2.scatter( outcat['CoordsOrig'][:,0], outcat['CoordsOrig'][:,1], s=12, c='k' )

    axbotright_mer2 = fig_mergers2.add_axes([0.55,0.05,w,w])
    axbotright_mer2.set_title('Downsampled mergers',fontsize=fs)
    axbotright_mer2.tick_params(axis='both',labelsize=fs)
    axbotright_mer2.xaxis.set_ticklabels(axbotright_mer2.get_xticklabels(),visible=False)
    axbotright_mer2.yaxis.set_ticklabels(axbotright_mer2.get_yticklabels(),visible=False)
    axbotright_mer2.set_xlim(axbotleft_mer2.get_xlim())
    axbotright_mer2.set_ylim(axbotleft_mer2.get_ylim())
    axbotright_mer2.scatter( (outcat['CoordsOrig'][:,0])[outcat['toomany_mergers-2']==0], (outcat['CoordsOrig'][:,1])[outcat['toomany_mergers-2']==0], s=12, c='k' )

    fig_mergers2.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers-2.pdf')

    print 'Number of particles remaining after downsampling mergers(2):',len(outcat['particleIDs'][outcat['toomany_mergers-2']==0])



    fig_mergers3 = plt.figure(figsize=(12,12))

    axtopleft_mer3 = fig_mergers3.add_axes([0.05,0.55,w,w])
    axtopleft_mer3.set_title('All ex-situ stellar particles',fontsize=fs)
    axtopleft_mer3.tick_params(axis='both',labelsize=fs)
    axtopleft_mer3.xaxis.set_ticklabels(axtopleft_mer3.get_xticklabels(),visible=False)
    axtopleft_mer3.yaxis.set_ticklabels(axtopleft_mer3.get_yticklabels(),visible=False)
    axtopleft_mer3.scatter( (outcat['CoordsOrig'][:,0])[outcat['is_insitu']==0], (outcat['CoordsOrig'][:,1])[outcat['is_insitu']==0], s=12, c='k' )

    axtopright_mer3 = fig_mergers3.add_axes([0.55,0.55,w,w])
    axtopright_mer3.set_title('Downsampled mergers',fontsize=fs)
    axtopright_mer3.tick_params(axis='both',labelsize=fs)
    axtopright_mer3.xaxis.set_ticklabels(axtopright_mer3.get_xticklabels(),visible=False)
    axtopright_mer3.yaxis.set_ticklabels(axtopright_mer3.get_yticklabels(),visible=False)
    axtopright_mer3.set_xlim(axtopleft_mer3.get_xlim())
    axtopright_mer3.set_ylim(axtopleft_mer3.get_ylim())
    axtopright_mer3.scatter( (outcat['CoordsOrig'][:,0])[(outcat['toomany_mergers-3']==0) & (outcat['is_insitu']==0)], 
                            (outcat['CoordsOrig'][:,1])[(outcat['toomany_mergers-3']==0) & (outcat['is_insitu']==0)], s=12, c='k' )

    axbotleft_mer3 = fig_mergers3.add_axes([0.05,0.05,w,w])
    axbotleft_mer3.set_title('All stellar particles',fontsize=fs)
    axbotleft_mer3.tick_params(axis='both',labelsize=fs)
    axbotleft_mer3.xaxis.set_ticklabels(axbotleft_mer3.get_xticklabels(),visible=False)
    axbotleft_mer3.yaxis.set_ticklabels(axbotleft_mer3.get_yticklabels(),visible=False)
    axbotleft_mer3.scatter( outcat['CoordsOrig'][:,0], outcat['CoordsOrig'][:,1], s=12, c='k' )

    axbotright_mer3 = fig_mergers3.add_axes([0.55,0.05,w,w])
    axbotright_mer3.set_title('Downsampled mergers',fontsize=fs)
    axbotright_mer3.tick_params(axis='both',labelsize=fs)
    axbotright_mer3.xaxis.set_ticklabels(axbotright_mer3.get_xticklabels(),visible=False)
    axbotright_mer3.yaxis.set_ticklabels(axbotright_mer3.get_yticklabels(),visible=False)
    axbotright_mer3.set_xlim(axbotleft_mer3.get_xlim())
    axbotright_mer3.set_ylim(axbotleft_mer3.get_ylim())
    axbotright_mer3.scatter( (outcat['CoordsOrig'][:,0])[outcat['toomany_mergers-3']==0], (outcat['CoordsOrig'][:,1])[outcat['toomany_mergers-3']==0], s=12, c='k' )

    fig_mergers3.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers-3.pdf')

    print 'Number of particles remaining after downsampling mergers(3):',len(outcat['particleIDs'][outcat['toomany_mergers-3']==0])


    fig_mergers4 = plt.figure(figsize=(12,12))

    axtopleft_mer4 = fig_mergers4.add_axes([0.05,0.55,w,w])
    axtopleft_mer4.set_title('All ex-situ stellar particles',fontsize=fs)
    axtopleft_mer4.tick_params(axis='both',labelsize=fs)
    axtopleft_mer4.xaxis.set_ticklabels(axtopleft_mer3.get_xticklabels(),visible=False)
    axtopleft_mer4.yaxis.set_ticklabels(axtopleft_mer3.get_yticklabels(),visible=False)
    axtopleft_mer4.scatter( (outcat['CoordsOrig'][:,0])[outcat['is_insitu']==0], (outcat['CoordsOrig'][:,1])[outcat['is_insitu']==0], s=12, c='k' )

    axtopright_mer4 = fig_mergers4.add_axes([0.55,0.55,w,w])
    axtopright_mer4.set_title('Redistributed mergers',fontsize=fs)
    axtopright_mer4.tick_params(axis='both',labelsize=fs)
    axtopright_mer4.xaxis.set_ticklabels(axtopright_mer3.get_xticklabels(),visible=False)
    axtopright_mer4.yaxis.set_ticklabels(axtopright_mer3.get_yticklabels(),visible=False)
    axtopright_mer4.set_xlim(axtopleft_mer3.get_xlim())
    axtopright_mer4.set_ylim(axtopleft_mer3.get_ylim())
    axtopright_mer4.scatter( (outcat['CoordsRedistributedMergers_z'+str(z)+'_f0.5'][:,0])[outcat['is_insitu']==0], 
                             (outcat['CoordsRedistributedMergers_z'+str(z)+'_f0.5'][:,1])[outcat['is_insitu']==0], s=12, c='k' )

    axbotleft_mer4 = fig_mergers4.add_axes([0.05,0.05,w,w])
    axbotleft_mer4.set_title('All stellar particles',fontsize=fs)
    axbotleft_mer4.tick_params(axis='both',labelsize=fs)
    axbotleft_mer4.xaxis.set_ticklabels(axbotleft_mer3.get_xticklabels(),visible=False)
    axbotleft_mer4.yaxis.set_ticklabels(axbotleft_mer3.get_yticklabels(),visible=False)
    axbotleft_mer4.scatter( outcat['CoordsOrig'][:,0], outcat['CoordsOrig'][:,1], s=12, c='k' )

    axbotright_mer4 = fig_mergers4.add_axes([0.55,0.05,w,w])
    axbotright_mer4.set_title('Redistributed mergers',fontsize=fs)
    axbotright_mer4.tick_params(axis='both',labelsize=fs)
    axbotright_mer4.xaxis.set_ticklabels(axbotright_mer3.get_xticklabels(),visible=False)
    axbotright_mer4.yaxis.set_ticklabels(axbotright_mer3.get_yticklabels(),visible=False)
    axbotright_mer4.set_xlim(axbotleft_mer3.get_xlim())
    axbotright_mer4.set_ylim(axbotleft_mer3.get_ylim())
    axbotright_mer4.scatter( (outcat['CoordsRedistributedMergers_z'+str(z)+'_f0.5'][:,0]), (outcat['CoordsRedistributedMergers_z'+str(z)+'_f0.5'][:,1]), s=12, c='k' )

    fig_mergers4.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers-4.pdf')


    #'CoordsRedistributedMergers_z'+str(z)+'_sq'


    Nbins = 80
    tx,ty0,ty1,ty2,ty3,ty4,ty5 = 5,4e4,2.5e4,1.5e4,0.9e4,0.5e4,0.3e4


    fig_distdem = plt.figure(figsize=(6,6))

    axdist_dem = fig_distdem.add_axes([0.15,0.15,0.75,0.75])
    axdist_dem.set_xlabel('Distance from subhalo center [kpc]')
    axdist_dem.set_ylabel('N particles')
    axdist_dem.set_yscale('log')
    axdist_dem.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_dem.text(tx, ty1, 'After restricting radial migration', fontsize=fs, color='orangered')
    axdist_dem.text(tx, ty2, '(In situ particles only)', fontsize=fs, color='orange')
    axdist_dem.axvline(2*ptclcat['Re_99'], color='grey', ls=':')
    axdist_dem.axvline(5*ptclcat['Re_99'], color='grey', ls=':')
    axdist_dem.set_ylim(1e0,1e5)
    axdist_dem.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_dem.hist(np.sqrt(outcat['DistDemigrated_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )
    axdist_dem.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==1], range=(0,Nbins), bins=Nbins, histtype='step', color='grey', alpha=0.8 )
    axdist_dem.hist(np.sqrt(outcat['DistDemigrated_sq'])[outcat['is_insitu']==1], range=(0,Nbins), bins=Nbins, histtype='step', color='orange', alpha=0.8 )

    fig_distdem.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_demigrated_distances.pdf')



    fig_distmer1 = plt.figure(figsize=(6,6))

    axdist_mer1 = fig_distmer1.add_axes([0.15,0.15,0.75,0.75])
    axdist_mer1.set_xlabel('Distance from subhalo center [kpc]')
    axdist_mer1.set_ylabel('N particles')
    axdist_mer1.set_yscale('log')
    axdist_mer1.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_mer1.text(tx, ty1, 'After downsampling mergers', fontsize=fs, color='orangered')
    axdist_mer1.text(tx, ty2, '(Ex situ particles only)', fontsize=fs, color='orange')
    axdist_mer1.axvline(5*ptclcat['Re_99'], color='grey', ls=':')
    axdist_mer1.set_ylim(1e0,1e5)
    axdist_mer1.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_mer1.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['toomany_mergers-1']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )
    axdist_mer1.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='grey', alpha=0.8 )
    axdist_mer1.hist(np.sqrt(outcat['DistOrig_sq'])[(outcat['is_insitu']==0) & (outcat['toomany_mergers-1']==0)], range=(0,Nbins), bins=Nbins, histtype='step', color='orange', alpha=0.8 )

    fig_distmer1.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers_distances-1.pdf')


    fig_distmer2 = plt.figure(figsize=(6,6))

    axdist_mer2 = fig_distmer2.add_axes([0.15,0.15,0.75,0.75])
    axdist_mer2.set_xlabel('Distance from subhalo center [kpc]')
    axdist_mer2.set_ylabel('N particles')
    axdist_mer2.set_yscale('log')
    axdist_mer2.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_mer2.text(tx, ty1, 'After downsampling mergers', fontsize=fs, color='orangered')
    axdist_mer2.text(tx, ty2, '(Ex situ particles only)', fontsize=fs, color='orange')
    axdist_mer2.axvline(5*ptclcat['Re_99'], color='grey', ls=':')
    axdist_mer2.set_ylim(1e0,1e5)
    axdist_mer2.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_mer2.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['toomany_mergers-2']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )
    axdist_mer2.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='grey', alpha=0.8 )
    axdist_mer2.hist(np.sqrt(outcat['DistOrig_sq'])[(outcat['is_insitu']==0) & (outcat['toomany_mergers-2']==0)], range=(0,Nbins), bins=Nbins, histtype='step', color='orange', alpha=0.8 )

    fig_distmer2.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers_distances-2.pdf')



    fig_distmer3 = plt.figure(figsize=(6,6))

    axdist_mer3 = fig_distmer3.add_axes([0.15,0.15,0.75,0.75])
    axdist_mer3.set_xlabel('Distance from subhalo center [kpc]')
    axdist_mer3.set_ylabel('N particles')
    axdist_mer3.set_yscale('log')
    axdist_mer3.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_mer3.text(tx, ty1, 'After downsampling mergers', fontsize=fs, color='orangered')
    axdist_mer3.text(tx, ty2, '(Ex situ particles only)', fontsize=fs, color='orange')
    axdist_mer3.axvline(5*ptclcat['Re_99'], color='grey', ls=':')
    axdist_mer3.set_ylim(1e0,1e5)
    axdist_mer3.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_mer3.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['toomany_mergers-3']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )
    axdist_mer3.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='grey', alpha=0.8 )
    axdist_mer3.hist(np.sqrt(outcat['DistOrig_sq'])[(outcat['is_insitu']==0) & (outcat['toomany_mergers-3']==0)], range=(0,Nbins), bins=Nbins, histtype='step', color='orange', alpha=0.8 )

    fig_distmer3.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers_distances-3.pdf')


    fig_distmer4_05 = plt.figure(figsize=(6,6))

    axdist_mer4_05 = fig_distmer4_05.add_axes([0.15,0.15,0.75,0.75])
    axdist_mer4_05.set_xlabel('Distance from subhalo center [kpc]')
    axdist_mer4_05.set_ylabel('N particles')
    axdist_mer4_05.set_yscale('log')
    axdist_mer4_05.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_mer4_05.text(tx, ty1, 'After redistributing z<20', fontsize=fs, color='orangered')
    axdist_mer4_05.text(tx, ty2, 'After redistributing z<2', fontsize=fs, color='mediumblue')
    axdist_mer4_05.text(tx, ty3, 'After redistributing z<1', fontsize=fs, color='darkviolet')
    axdist_mer4_05.text(tx, ty4, '(Ex situ particles only)', fontsize=fs, color='orange')
    axdist_mer4_05.axvline(5*ptclcat['Re_99'], color='grey', ls=':')
    axdist_mer4_05.set_ylim(1e0,1e5)
    axdist_mer4_05.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_mer4_05.hist(np.sqrt(outcat['DistRedistributedMergers_z20_f0.5_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )
    axdist_mer4_05.hist(np.sqrt(outcat['DistRedistributedMergers_z2_f0.5_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='mediumblue' )
    axdist_mer4_05.hist(np.sqrt(outcat['DistRedistributedMergers_z1_f0.5_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='darkviolet' )

    axdist_mer4_05.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='grey', alpha=0.8 )
    axdist_mer4_05.hist(np.sqrt(outcat['DistRedistributedMergers_z20_f0.5_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered', alpha=0.7 )
    axdist_mer4_05.hist(np.sqrt(outcat['DistRedistributedMergers_z2_f0.5_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='mediumblue', alpha=0.7 )
    axdist_mer4_05.hist(np.sqrt(outcat['DistRedistributedMergers_z1_f0.5_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='darkviolet', alpha=0.7 )


    fig_distmer4_05.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers_distances-4_f0.5.pdf')


    fig_distmer4_07 = plt.figure(figsize=(6,6))

    axdist_mer4_07 = fig_distmer4_07.add_axes([0.15,0.15,0.75,0.75])
    axdist_mer4_07.set_xlabel('Distance from subhalo center [kpc]')
    axdist_mer4_07.set_ylabel('N particles')
    axdist_mer4_07.set_yscale('log')
    axdist_mer4_07.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_mer4_07.text(tx, ty1, 'After redistributing z<20', fontsize=fs, color='orangered')
    axdist_mer4_07.text(tx, ty2, 'After redistributing z<2', fontsize=fs, color='mediumblue')
    axdist_mer4_07.text(tx, ty3, 'After redistributing z<1', fontsize=fs, color='darkviolet')
    axdist_mer4_07.text(tx, ty4, '(Ex situ particles only)', fontsize=fs, color='orange')
    axdist_mer4_07.axvline(5*ptclcat['Re_99'], color='grey', ls=':')
    axdist_mer4_07.set_ylim(1e0,1e5)
    axdist_mer4_07.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    #axdist_mer4_07.hist(np.sqrt(outcat['DistRedistributedMergers_z20_f0.7_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )
    axdist_mer4_07.hist(np.sqrt(outcat['DistRedistributedMergers_z2_f0.7_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='mediumblue' )
    #axdist_mer4_07.hist(np.sqrt(outcat['DistRedistributedMergers_z1_f0.7_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='darkviolet' )

    axdist_mer4_07.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='grey', alpha=0.8 )
    #axdist_mer4_07.hist(np.sqrt(outcat['DistRedistributedMergers_z20_f0.7_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered', alpha=0.7 )
    axdist_mer4_07.hist(np.sqrt(outcat['DistRedistributedMergers_z2_f0.7_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='mediumblue', alpha=0.7 )
    #xaxdist_mer4_07.hist(np.sqrt(outcat['DistRedistributedMergers_z1_f0.7_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='darkviolet', alpha=0.7 )


    fig_distmer4_07.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers_distances-4_f0.7.pdf')




    fig_distinsitu = plt.figure(figsize=(6,6))

    axdist_insitu = fig_distinsitu.add_axes([0.15,0.15,0.75,0.75])
    axdist_insitu.set_xlabel('Distance from subhalo center [kpc]')
    axdist_insitu.set_ylabel('N particles')
    axdist_insitu.set_yscale('log')
    axdist_insitu.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_insitu.text(tx, ty1, 'Ex-situ particles only', fontsize=fs, color='orangered')
    axdist_insitu.axvline(5*ptclcat['Re_99'], color='grey', ls=':')
    axdist_insitu.set_ylim(1e0,1e5)
    axdist_insitu.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_insitu.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )

    fig_distinsitu.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_insitu_distances.pdf')




    fig_distpost = plt.figure(figsize=(6,6))

    axdist_post = fig_distpost.add_axes([0.15,0.15,0.75,0.75])
    axdist_post.set_xlabel('Distance from subhalo center [kpc]')
    axdist_post.set_ylabel('N particles')
    axdist_post.set_yscale('log')
    axdist_post.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_post.text(tx, ty1, 'After removing ex-situ, post-infall particles', fontsize=fs, color='orangered')
    axdist_post.axvline(5*ptclcat['Re_99'], color='grey', ls=':')
    axdist_post.set_ylim(1e0,1e5)
    axdist_post.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_post.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_exsitu_postinfall']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )


    fig_distpost.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_postinfall_distances.pdf')


    


    fig_checkmergers = plt.figure(figsize=(6,6))

    axcheck = fig_checkmergers.add_axes([0.15,0.15,0.75,0.75])
    axcheck.set_xlabel('snap_strip')
    axcheck.set_ylabel('log(progenitor galaxy stellar mass)')
    axcheck.set_ylim(6,11)
    axcheck.plot(ptclcat['snap_strip'][outcat['is_insitu']==1],ptclcat['log_mstell_strip'][outcat['is_insitu']==1],'o',mfc='k',mec='none',alpha=1.)
    axcheck.plot(ptclcat['snap_strip'][outcat['is_insitu']==0],ptclcat['log_mstell_strip_prev'][outcat['is_insitu']==0],'o',mfc='grey',mec='none',alpha=0.8)

    print ptclcat['log_mstell_strip'][outcat['is_insitu']==1]
    print len(ptclcat['log_mstell_strip'][outcat['is_insitu']==1])
    print np.min(ptclcat['log_mstell_strip'][outcat['is_insitu']==1]),np.max(ptclcat['log_mstell_strip'][outcat['is_insitu']==1])

    axcheck.plot(ptclcat['snap_strip'][outcat['toomany_mergers-3']==3],ptclcat['log_mstell_strip_prev'][outcat['toomany_mergers-3']==3],'o',mfc='purple',mec='none',alpha=0.8)
    axcheck.plot(ptclcat['snap_strip'][outcat['toomany_mergers-3']==2],ptclcat['log_mstell_strip_prev'][outcat['toomany_mergers-3']==2],'o',mfc='blue',mec='none',alpha=0.8)
    axcheck.plot(ptclcat['snap_strip'][outcat['toomany_mergers-3']==1],ptclcat['log_mstell_strip_prev'][outcat['toomany_mergers-3']==1],'o',mfc='green',mec='none',alpha=0.8)

    
    print 'checking..'
    print len(ptclcat['log_mstell_strip_prev'][outcat['toomany_mergers-3']==1])
    print len(ptclcat['log_mstell_strip_prev'][outcat['toomany_mergers-3']==2])
    print len(ptclcat['log_mstell_strip_prev'][outcat['toomany_mergers-3']==3])

    fig_checkmergers.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_mergercheck.pdf')



    fig_distmer_m5 = plt.figure(figsize=(6,6))

    axdist_mer_m5 = fig_distmer_m5.add_axes([0.15,0.15,0.75,0.75])
    axdist_mer_m5.set_xlabel('Distance from subhalo center [kpc]')
    axdist_mer_m5.set_ylabel('N particles')
    axdist_mer_m5.set_yscale('log')
    axdist_mer_m5.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_mer_m5.text(tx, ty1, 'After downsampling mergers', fontsize=fs, color='orangered')
    axdist_mer_m5.text(tx, ty2, '(Ex situ particles only)', fontsize=fs, color='orange')
    axdist_mer_m5.set_ylim(1e0,1e5)
    axdist_mer_m5.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_mer_m5.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['toomany_mergers-5']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )
    axdist_mer_m5.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='grey', alpha=0.8 )
    axdist_mer_m5.hist(np.sqrt(outcat['DistOrig_sq'])[(outcat['is_insitu']==0) & (outcat['toomany_mergers-5']==0)], range=(0,Nbins), bins=Nbins, histtype='step', color='orange', alpha=0.8 )

    fig_distmer_m5.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers_distances-m5.pdf')




    fig_distmer_m6 = plt.figure(figsize=(6,6))

    axdist_mer_m6 = fig_distmer_m6.add_axes([0.15,0.15,0.75,0.75])
    axdist_mer_m6.set_xlabel('Distance from subhalo center [kpc]')
    axdist_mer_m6.set_ylabel('N particles')
    axdist_mer_m6.set_yscale('log')
    axdist_mer_m6.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_mer_m6.text(tx, ty1, 'After downsampling mergers', fontsize=fs, color='orangered')
    axdist_mer_m6.text(tx, ty2, '(Ex situ particles only)', fontsize=fs, color='orange')
    axdist_mer_m6.set_ylim(1e0,1e5)
    axdist_mer_m6.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_mer_m6.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['toomany_mergers-6']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )
    #axdist_mer_m6.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='grey', alpha=0.8 )
    #axdist_mer_m6.hist(np.sqrt(outcat['DistOrig_sq'])[(outcat['is_insitu']==0) & (outcat['toomany_mergers-6']==0)], range=(0,Nbins), bins=Nbins, histtype='step', color='orange', alpha=0.8 )

    fig_distmer_m6.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers_distances-m6.pdf')




    fig_distmer_m7 = plt.figure(figsize=(6,6))

    axdist_mer_m7 = fig_distmer_m7.add_axes([0.15,0.15,0.75,0.75])
    axdist_mer_m7.set_xlabel('Distance from subhalo center [kpc]')
    axdist_mer_m7.set_ylabel('N particles')
    axdist_mer_m7.set_yscale('log')
    axdist_mer_m7.text(tx, ty0, 'Original profile', fontsize=fs,color='k')
    axdist_mer_m7.text(tx, ty1, 'After downsampling mergers', fontsize=fs, color='orangered')
    axdist_mer_m7.text(tx, ty2, '(Ex situ particles only)', fontsize=fs, color='orange')
    axdist_mer_m7.set_ylim(1e0,1e5)
    axdist_mer_m7.hist(np.sqrt(outcat['DistOrig_sq']), range=(0,Nbins), bins=Nbins, histtype='step', color='k' )
    axdist_mer_m7.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['toomany_mergers-7']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='orangered' )
    axdist_mer_m7.hist(np.sqrt(outcat['DistOrig_sq'])[outcat['is_insitu']==0], range=(0,Nbins), bins=Nbins, histtype='step', color='grey', alpha=0.8 )
    axdist_mer_m7.hist(np.sqrt(outcat['DistOrig_sq'])[(outcat['is_insitu']==0) & (outcat['toomany_mergers-7']==0)], range=(0,Nbins), bins=Nbins, histtype='step', color='orange', alpha=0.8 )

    fig_distmer_m7.savefig(plotsdir+'/adjusted_subhalo'+str(subfindID)+'_downsampledmergers_distances-m7.pdf')



    return None



def write_adjusted_particles(outcat):
    h5f = h5py.File(outdir+'/adjusted_subhalo'+str(subfindID)+'.hdf5','w')

    for keyname in outcat.keys():
        h5f.create_dataset(keyname,data=outcat[keyname],compression='gzip',compression_opts=9,chunks=True)
    h5f.close()
    return None

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

    'Mode options'
    verbose = arguments['--verbose']

    if verbose:
        print '\n',arguments,'\n'

    'Directory structure'
    TNGdir = arguments['--TNGdir']
    ptcldir = arguments['--ptcldir']
    plotsdir = arguments['--plotsdir']
    outdir = arguments['--outdir']
    mkdirp(plotsdir)
    mkdirp(outdir)

    'Simulation details'
    snapnum = int(arguments['--snapnum'])
    h = np.float(arguments['--littleh'])
    a = grab_scalefactor(snapnum)


    myTNG = TNGdir.split('/')[-2]
    check_TNG()

    'Subhalo of interest'
    subfindID = int(arguments['<subfindID>'])

    'Read in subhalo particle data / standard TNG outputs'
    'Coordinates are in ckpc/h'
    subhalostarsandwinds = il.snapshot.loadSubhalo(TNGdir+'/output/',snapnum,subfindID,'stars')
    check_for_stars(subhalostarsandwinds)

    'Conversions'
    conver = ascii.read('snapnumbers.txt')
    map_snap2a = dict(zip(conver['Snap'],conver['Scalefactor']))
    map_z2snap = dict(zip(conver['Redshift'],conver['Snap']))

    'Read in particle tracking data'
    ptclcat = read_particle_tracker_outputs(subfindID)

    'Set up outputs'
    if verbose:
        print_verbose_string('Creating output catalog and copying over subhalo Coordinates ...')
    outcat = {}
    outcat['particleIDs'] = np.array(ptclcat['particleIDs'])

    'Fill in coordinates (important because outcat and ptclcat are in the same order, but subhalocat is NOT)'
    outcat = fill_coords(outcat,subhalostarsandwinds)


    'ID in-situ stellar particles (all)'
    if verbose:
        print_verbose_string('Flagging all in-situ particles ...')
    outcat = find_insitu(outcat,ptclcat)
    outcat = contract_insitu(outcat,ptclcat,subhalostarsandwinds,freduce=0.6)


    'ID ex-situ stellar particles that formed post-infall'
    if verbose:
        print_verbose_string('Flagging ex-situ, post-infall stellar particles ...')
    outcat = find_exsitu_postinfall(outcat,ptclcat)


    'ID and de-migrate in-situ stellar particles that moved too far outwards'
    if verbose:
        print_verbose_string('Flagging in-situ  particles that migrated out beyond 5Re and "de-migrating" them back to smaller radii ...')
    outcat = find_insitu_demigrate(outcat,ptclcat,subhalostarsandwinds)

    'ID and down-sample ex-situ stellar particles from a given mass and redshift range'
    if verbose:
        print_verbose_string('Flagging ex-situ particles with a specified range of mprog and zacc and downsampling ...')
    'mergers-1'
    outcat = find_exsitu_massivemergers_uniform(outcat,ptclcat)

    'some other stuff'
    outcat = find_exsitu_massivemergers_z1(outcat,ptclcat)
    outcat = find_exsitu_massivemergers_mostmassive(outcat,ptclcat)
    outcat = find_exsitu_mergers_redistribute(outcat,ptclcat,subhalostarsandwinds,z=20,freduce=0.5)
    outcat = find_exsitu_mergers_redistribute(outcat,ptclcat,subhalostarsandwinds,z=2,freduce=0.5)

    'mergers-4'
    outcat = find_exsitu_mergers_redistribute(outcat,ptclcat,subhalostarsandwinds,z=2,freduce=0.7)

    'meh'
    outcat = find_exsitu_mergers_redistribute(outcat,ptclcat,subhalostarsandwinds,z=1,freduce=0.5)

    'mergers-5'
    outcat = find_exsitu_massivemergers_remove_z2(outcat,ptclcat)

    'mergers-6'
    outcat = find_exsitu_massivemergers_remove_all(outcat,ptclcat)

    'mergers-7'
    outcat = find_exsitu_downsample_individual(outcat,ptclcat)

    'PLOT VISUALIZATION OF OUTPUTS'
    if verbose:
        print_verbose_string('Generating some diagnostic plots ...')
    plot_adjusted_particles(outcat,subhalostarsandwinds)

    'SAVE OUTPUTS'
    if verbose:
        print_verbose_string('Writing output to an hdf5 file ...')
    write_adjusted_particles(outcat)


    print '\nDone.\n'
