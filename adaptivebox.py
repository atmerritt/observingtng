#!/usr/bin/env python
""" adaptivebox.py -- Given particle data for a TNG subhalo, create data boxes with the 3D adaptively-smoothed distribution of either mass or light (in a given band).

Usage:
    adaptivebox.py [-h] [-v] [-O] [-g] [-r] [-i] [-m] [--adjustment TYPE] [--projection TYPE] [--distance MPC] [--pixelsize KPC] [--kernelsize INT] [--multkernel INT] [--littleh FLT] [--snapnum INT] [--TNGdir DIR] [--adjustdir DIR] [--plotsdir DIR] [--boxdir DIR] [--subhaloID INT] [--filename STR]

Options:
    -h, --help                    Show this screen                                                                          [default: False]
    -v, --verbose                 Print extra information                                                                   [default: False]
    -O, --overwrite               Overwrite existing boxes                                                                  [default: False]

    -g, --gband                   Create box with distribution of SDSS g (AB) flux                                          [default: False]
    -r, --rband                   Create box with distribution of SDSS r (AB) flux                                          [default: False]
    -i, --iband                   Create box with distribution of SDSS i (AB) flux                                          [default: False]
    -m, --mass                    Create box with distribution of stellar mass                                              [default: False]

    -j TYPE, --adjustment TYPE    "exsitu","insitucontract", "postinfall", "demigrate", "mergers-[12367]" or "mergers-4-z[1,2,20]-f0[57]"      [default: None]
    -p TYPE, --projection TYPE    Optionally specify a Faceon or Edgeon image                                               [default: None]
    -d MPC, --distance MPC        Distance to place galaxy at (in Mpc)                                                      [default: 10.]
    -x KPC, --pixelsize KPC       Pixel sizes in kpc                                                                        [default: 0.25]
    -k INT, --kernelsize INT      Smooth particles with kernel sizes corresponding to the kth nearest neighbor              [default: 3]
    -R INT, --multkernel INT      Smooth particles over pixels within R*kernelsize                                          [default: 5]

    -l FLT, --littleh FLT         Value of "little h"                                                                       [default: 0.6774]
    -s INT, --snapnum INT         Snap Number (99 corresponds to z=0)                                                       [default: 99]

    -t DIR, --TNGdir DIR          Directory where TNG outputs live                                                          [default: /virgo/data/IllustrisTNG/L75n1820TNG/]
    -a DIR, --adjustdir DIR       Directory where "adjusted" particle information lives                                     [default: /u/allim/Projects/TNG/Subhalos/adjustments/L75n1820TNG/]
    -o DIR, --plotsdir DIR        Directory where diagnostic plots will live                                                [default: /u/allim/Projects/TNG/Plots/boxes/]
    -c DIR, --boxdir DIR          Directory where boxes will live                                                           [default: /u/allim/Projects/TNG/Boxes/]

    -b INT, --subhaloID INT       SubhaloID to create a box / boxes for
    -f STR, --filename STR        File containing a list of SubhaloIDs to create boxes for

Example:
    python adaptivebox.py -v -m --subhaloID 505333
    python adaptivebox.py -v -m -g -r --subhaloID 505333 
    python adaptivebox.py -v -m -t /ptmp/allim/TNG/Snapshot99/ -x 0.25 --subhaloID 505333 (batch mode)
"""

from __future__ import division
from memory_profiler import profile

import illustris_python as il
import tng_tools as tt

import matplotlib as ml
ml.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc,colorbar,lines
import pylab

from scipy.spatial import cKDTree as KDT
from astropy.io import fits,ascii
from astropy.table import Table
import astropy.units as u
from numpy import sqrt as npsqrt

import numpy as np
import itertools
import datetime
import numexpr
import seaborn
import docopt
import pylab
import h5py
import math
import sys
import os
import re
import gc

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


def check_size(boxshape):
    if verbose:
        print_verbose_string('Checking boxsize: ('+str(boxshape[0])+','+str(boxshape[1])+','+str(boxshape[2])+')')

    if boxshape[0] > 2000 or boxshape[1] > 2000 or boxshape[2] > 2000:
        print '\n*** Size Error: This box is too big! Rerun with larger pixels or something. ***\n'
        sys.exit()
    return None

def check_input(arguments):
    if (arguments['--subhaloID'] is None) and (arguments['--filename'] is None):
        print '\n*** Input Error: You have to specify either a subhaloID or a filename containing multiple subhaloIDs! Try again. ***\n'
        sys.exit()

    if (arguments['--subhaloID'] is not None) and (arguments['--filename'] is not None):
        print '\n*** Input Error: You have to specify either a subhaloID _OR_ a filename containing multiple subhaloIDs (not both)! Try again. ***\n'
        sys.exit()

    allowable_adjustments = ['None',
                             'exsitu','insitucontract','postinfall','demigrate',
                             'mergers-1','mergers-2','mergers-3',
                             'mergers-4-z1-f05','mergers-4-z2-f05','mergers-4-z20-f05',
                             'mergers-4-z2-f07','mergers-6','mergers-7']

    if arguments['--adjustment'] not in allowable_adjustments:
        print '\n*** Input Error: Are you trying to specify an adjustment? The options are:'
        print allowable_adjustments
        print 'Try again. ***\n'

    return None

def consolidate_output_choices():
    cat = {}
    catkeys = ['normkey','photind','name']
    for catkey in catkeys:
        cat[catkey] = []

    if mass:
        cat['normkey'].append('Masses')
        cat['photind'].append(999)
        cat['name'].append('StellarMass')

    if sdss_g:
        cat['normkey'].append('GFM_StellarPhotometrics')
        cat['photind'].append(4)
        cat['name'].append('gband')

    if sdss_r:
        cat['normkey'].append('GFM_StellarPhotometrics')
        cat['photind'].append(5)
        cat['name'].append('rband')

    if sdss_i:
        cat['normkey'].append('GFM_StellarPhotometrics')
        cat['photind'].append(6)
        cat['name'].append('iband')

    if len(cat['name']) == 0:
        print '\n *** Input Error: Which particle property do you care about? Pick at least one of: stellar mass, gband, rband, iband. Try again. ***\n'
        sys.exit()

    return cat


def grab_scalefactor(snapnum):
    if not os.path.isfile('snapnumbers.txt'):
        print '\n**** Where is snapnumbers.txt?? ****\n'
        sys.exit()

    conver = ascii.read('snapnumbers.txt')
    a = np.float(conver['Scalefactor'][conver['Snap']==snapnum])
    return a

def make_hist_2d(subhalostars,subhalo_tag,normkey='Masses',photind=None,distance=10.,xbins=None,ybins=None,outdir=None):

    'Choose how to weight distribution'
    if normkey == 'Masses':
        weights = subhalostars[normkey]*(1e10)/h
    else:
        weights = get_scaled_flux(subhalostars['GFM_StellarPhotometrics'][:,photind],distance)

    'Make histogram'
    h2d,xe,ye = np.histogram2d(subhalostars['CoordinatesOrigin'][:,0]*(a/h),subhalostars['CoordinatesOrigin'][:,1]*(a/h),bins=(xbins,ybins),weights=weights)

    h5 = h5py.File(outdir+'/'+subhalo_tag+'_snap'+str(snapnum)+'_hist2d.hdf5','w')
    h5.create_dataset('h2d',data=h2d,compression='gzip',compression_opts=9,chunks=True)
    h5.close()
    return None



def save_outputs(subhaloID,subhalo_tag,outdir,data=None,sig=None,maxr=None,x=None,y=None,z=None,disttype=None,dustmodel='None',dist=10.,pixkpc=0.25,proj='None',littleh=0.6774,snapnum=99):

    'For space, we will only save the 3 projections of this box.'
    box_xy = np.sum(data,axis=2)
    box_yz = np.sum(data,axis=0)
    box_xz = np.sum(data,axis=1)

    'Units!!'
    if 'Mass' in subhalo_tag:
        counts_units = 'Msun'
    else:
        counts_units = '10**24 erg/s/cm**2/Hz'

    'Do saving things here ..'
    h5fe = h5py.File(outdir+'/'+subhalo_tag+'_snap'+str(snapnum)+'.hdf5','w')
    h5fe.create_dataset('box_xy',data=box_xy,compression='gzip',compression_opts=9,chunks=True)
    h5fe.create_dataset('box_yz',data=box_yz,compression='gzip',compression_opts=9,chunks=True)
    h5fe.create_dataset('box_xz',data=box_xz,compression='gzip',compression_opts=9,chunks=True)
    h5fe.create_dataset('subhaloID',data=subhaloID)
    h5fe.create_dataset('ptclsig',data=sig,compression='gzip',compression_opts=9,chunks=True)
    h5fe.create_dataset('ptclmaxr',data=maxr,compression='gzip',compression_opts=9,chunks=True)
    h5fe.create_dataset('x',data=x,compression='gzip',compression_opts=9,chunks=True)
    h5fe.create_dataset('y',data=y,compression='gzip',compression_opts=9,chunks=True)
    h5fe.create_dataset('z',data=z,compression='gzip',compression_opts=9,chunks=True)
    h5fe.create_dataset('type',data=disttype)
    h5fe.create_dataset('dust',data=dustmodel)
    h5fe.create_dataset('distMpc',data=dist)
    h5fe.create_dataset('kpcperpixel',data=pixkpc)
    h5fe.create_dataset('projection',data=proj)
    h5fe.create_dataset('h',data=littleh)
    h5fe.create_dataset('snapnum',data=snapnum)
    h5fe.create_dataset('units_pixelcounts',data=counts_units)
    h5fe.close()

    'Clear out the mem file too'
    clearit(boxdir +'/'+myTNG+'/mems/subhalo'+str(subhaloID)+'_snap'+str(snapnum)+'.npmem')

    print_verbose_string('Output file saved to: '+outdir+'/'+subhalo_tag+'_snap'+str(snapnum)+'.hdf5')
    return None

def check_for_stars(subhalostarsandwinds):
    if 'ParticleIDs' not in subhalostarsandwinds.keys():
        print '\n*** Something went wrong - no star particles found for this subhalo. ***\n'
        sys.exit()
    return None

def exclude_wind_particles(subhalostarsandwinds):
    subhalostars = {}
    for keyname in subhalostarsandwinds.keys():
        if keyname == 'count':
            subhalostars[keyname] = len((subhalostarsandwinds['ParticleIDs'])[subhalostarsandwinds['GFM_StellarFormationTime'] > 0])
        else:
            subhalostars[keyname] = (subhalostarsandwinds[keyname])[subhalostarsandwinds['GFM_StellarFormationTime'] > 0]
    return subhalostars

@profile
def build_box(subhalostars,pixelsizekpc):
    'determine minmax of xyz coordinates'
    xmin,xmax = np.min(subhalostars['CoordinatesOrigin'][:,0])*(a/h),np.max(subhalostars['CoordinatesOrigin'][:,0])*(a/h)
    ymin,ymax = np.min(subhalostars['CoordinatesOrigin'][:,1])*(a/h),np.max(subhalostars['CoordinatesOrigin'][:,1])*(a/h)
    zmin,zmax = np.min(subhalostars['CoordinatesOrigin'][:,2])*(a/h),np.max(subhalostars['CoordinatesOrigin'][:,2])*(a/h)

    'make sure the grid is not too huge. (only an issue for the largest galaxies..)'
    # NOTE: dec4,2019 - temp. hard-coding limits to force the adjusted frames to have the same dimensions as the unadjusted frames.
    # this is an issue for the lower mass galaxies and results in messed up surface density profiles etc.
    max_extent = 100.
    xmin = -100. #np.max((xmin,-1*max_extent))
    xmax = 100. #np.min((xmax,max_extent))
    ymin = -100. #np.max((ymin,-1*max_extent))
    ymax = 100. #np.min((ymax,max_extent))
    zmin = -100. #np.max((zmin,-1*max_extent))
    zmax = 100. #np.min((zmax,max_extent))


    'create pixel grid, given pixelsize in kpc'
    buff = 0.
    xpix = np.arange(xmin-buff,xmax+buff+pixelsizekpc,pixelsizekpc)
    ypix = np.arange(ymin-buff,ymax+buff+pixelsizekpc,pixelsizekpc)
    zpix = np.arange(zmin-buff,zmax+buff+pixelsizekpc,pixelsizekpc)

    #'Check to make sure this box is not going to be too huge!'
    #check_size((xpix.shape[0],ypix.shape[0],zpix.shape[0]))

    if verbose:
        print_verbose_string('Building (empty) 3D grid ... --> size is: ('+str(xpix.shape[0])+','+str(ypix.shape[0])+','+str(zpix.shape[0])+')')

    #emptybox = np.zeros((xpix.shape[0],ypix.shape[0],zpix.shape[0]))
    emptybox = np.memmap(boxdir +'/'+myTNG+'/mems/subhalo'+str(subhaloID)+'_snap'+str(snapnum)+'.npmem',shape=(xpix.shape[0],ypix.shape[0],zpix.shape[0]),dtype='float64',mode='w+')
    testingsomething = 2.
    cx,cy,cz = np.meshgrid(xpix,ypix,zpix,indexing='ij',copy=False)

    return emptybox,cx,cy,cz

def calc_coord_shift(coo,pt,box_ckpc,maxsep=1000):
    'Note: both coordinates and the box size are given in ckpc/h'

    scoo = np.array(coo)

    if np.max(coo - pt) > maxsep:
        if verbose:
            print_verbose_string('Coordinate wrapping detected! Adjusting...')
            scoo[coo - pt > maxsep] = coo[coo - pt > maxsep] - box_ckpc

    elif np.max(pt - coo) > maxsep:
        if verbose:
            print_verbose_string('Coordinate wrapping detected! Adjusting...')
        scoo[pt - coo > maxsep] = coo[pt - coo > maxsep] + box_ckpc

    return scoo

def check_edge_effects(subhalostars):
    if verbose:
        print_verbose_string('Checking for edge effects...')

    'Set box size'
    box_cMpc = 75
    box_ckpc = (1e3)*box_cMpc

    'Find deepest potential'
    ox,oy,oz,ovx,ovy,ovz = tt.calc_angmom.find_deepest_potential(subhalostars)

    'Shift coordinates if necessary (if no wrapping has happened, Coordinates and CoordinatesShifted are the same thing)'
    sx = calc_coord_shift(subhalostars['Coordinates'][:,0],ox,box_ckpc,maxsep=box_ckpc/2)
    sy = calc_coord_shift(subhalostars['Coordinates'][:,1],oy,box_cMpc,maxsep=box_ckpc/2)
    sz = calc_coord_shift(subhalostars['Coordinates'][:,2],oz,box_cMpc,maxsep=box_ckpc/2)

    subhalostars['CoordinatesShifted'] = np.dstack((np.array(sx),np.array(sy),np.array(sz))).reshape((len(sx),3))

    return subhalostars


def make_adjustments(subhalostars,adjustment='None'):
    'If no adjustments are specified, we just copy over all the coordinates '
    'Note: CoordinatesShifted came from check_edge_effects, use this as the default value if no adjustments are to be made'
    subhalostars['CoordinatesAdjusted'] = np.array(subhalostars['CoordinatesShifted'])

    if adjustment != 'None':
        if verbose:
            print_verbose_string('Making adjustments! Type: '+adjustment)

        f = h5py.File(adjustdir+'/adjusted_subhalo'+str(subhaloID)+'.hdf5')
        adjusted_pid = f.get('particleIDs').value
        is_insitu = f.get('is_insitu').value
        is_postinfall = f.get('is_exsitu_postinfall').value
        toomany_mergers1 = f.get('toomany_mergers-1').value
        toomany_mergers2 = f.get('toomany_mergers-2').value
        toomany_mergers3 = f.get('toomany_mergers-3').value
        toomany_mergers6 = f.get('toomany_mergers-6').value
        toomany_mergers7 = f.get('toomany_mergers-7').value

        'Update CoordinatesAdjusted in subhalostars if necessary'
        if adjustment in ['insitucontract','demigrate','mergers-4-z20-f05','mergers-4-z2-f05','mergers-4-z1-f05','mergers-4-z2-f07']:
            if adjustment == 'demigrate':
                adjusted_coord = f.get('CoordsDemigrated').value
            elif adjustment == 'insitucontract':
                adjusted_coord = f.get('CoordsContractInSitu')
            elif adjustment in ['mergers-4-z20-f05','mergers-4-z2-f05','mergers-4-z1-f05','mergers-4-z2-f07']:
                lastint = adjustment[-1]
                coordkey = 'CoordsRedistributedMergers_z'+str(adjustment.split('-z')[-1].split('-')[0])+'_f0.'+str(lastint)
                adjusted_coord = f.get(coordkey).value
            for ind,pID in enumerate(subhalostars['ParticleIDs']):
                subhalostars['CoordinatesAdjusted'][ind,:] = adjusted_coord[adjusted_pid==pID,:]


        'Flag various things that we want to exclude'
        particle_flags = np.zeros(len(subhalostars['ParticleIDs']))
        if adjustment == 'combined':
            '1 if: is_insitu == 0, is_postinfall == 0, toomany_mergers-[123] == 0'
            for ind,pID in enumerate(subhalostars['ParticleIDs']):
                particle_flags[ind] = np.max(( np.float(is_insitu[adjusted_pid==pID]), np.float(is_postinfall[adjusted_pid==pID]), 
                                               np.float(toomany_mergers1[adjusted_pid==pID]), np.float(toomany_mergers2[adjusted_pid==pID]), np.float(toomany_mergers3[adjusted_pid==pID]) ))

        if adjustment in ['exsitu','postinfall','mergers-1','mergers-2','mergers-3','mergers-6','mergers-7']:
            if adjustment == 'exsitu':
                check_flag = is_insitu
            elif adjustment == 'postinfall':
                check_flag = is_postinfall
            elif adjustment == 'mergers-1':
                check_flag = toomany_mergers1
            elif adjustment == 'mergers-2':
                check_flag = toomany_mergers2
            elif adjustment == 'mergers-3':
                check_flag = toomany_mergers3
            elif adjustment == 'mergers-6':
                check_flag = toomany_mergers6
            elif adjustment == 'mergers-7':
                check_flag = toomany_mergers7

            for ind,pID in enumerate(subhalostars['ParticleIDs']):
                particle_flags[ind] = np.float(check_flag[adjusted_pid==pID])

        'Ignore flagged particles'
        if adjustment in ['exsitu','postinfall','mergers-1','mergers-2','mergers-3','mergers-6','mergers-7','combined']:
            for key in subhalostars.keys():
                if key != 'count':
                    subhalostars[key] = subhalostars[key][particle_flags==0.]

        f.close()

    else:
        if verbose:
            print_verbose_string('No adjustments needed! Carrying on ...')

    return subhalostars

def get_mini_indices(boxshape,pxi,pyi,pzi,pad):
    xii = np.max((0,pxi-pad))
    yii = np.max((0,pyi-pad))
    zii = np.max((0,pzi-pad))

    xif = np.min((boxshape[0],pxi+pad+1))
    yif = np.min((boxshape[1],pyi+pad+1))
    zif = np.min((boxshape[2],pzi+pad+1))

    return xii,xif,yii,yif,zii,zif

def get_scaled_flux(absmag,Mpc):
    'absolute magnitude to apparent magnitude (AB)'
    'distance modulus: m - M = 5log10(d/pc) - 5'
    m_AB = absmag + 5*np.log10(Mpc*(1e6)) - 5

    'AB mag to flux; now in units of erg/s/cm^2/Hz'
    'm_AB = -2.5*np.log10(flux) - 48.60'
    fluxes = 10**((m_AB + 48.60)/(-2.5))

    ### just until we work out file sizes! can undo this later, in the next stage
    fluxes *= 1e24

    return fluxes



def measure_particleparticle(subhalostars,emptybox,boxx,boxy,boxz,kneighbors=3,maxNkernels=5,pixelsizekpc=0.25,distance=10.,projection='None',outtag=None):
    if verbose:
        print_verbose_string('Measuring particle-particle distances in 3D ...')

    'Choose which key to use for projection'
    if projection == 'None':
        projkey = 'CoordinatesOrigin'
    else:
        projkey = 'Coordinates'+projection

    'Put particle coordinates in physical units'
    ptclx = subhalostars[projkey][:,0]*(a/h)
    ptcly = subhalostars[projkey][:,1]*(a/h)
    ptclz = subhalostars[projkey][:,2]*(a/h)
    ptcld2o = np.sqrt(ptclx**2 + ptcly**2 + ptclz**2)

    'Calculate distance to the kth nearest neighbor (star particles), and the physical extent of the smoothing'
    ptclcoords = zip(ptclx,ptcly,ptclz)
    ptclkdt = KDT(ptclcoords)
    ptcldistances_k,ids_k = ptclkdt.query(ptclcoords,k=kneighbors)
    rstell = ptcldistances_k[:,-1] + pixelsizekpc
    rstell[rstell > 30] = 30.
    maxrstell = maxNkernels*rstell

    'Plot kth nearest neighbor and smoothing extents for this galaxy'
    plot_kneighbors(ptclkdt,ptclcoords,outtag,pixelsizekpc)
    plot_smoothing_extent(maxrstell,outtag,pixelsizekpc)

    'Place each particle in its pixel bin'
    'NOTE: the xyz values in the box[xyz] arrays are *not* pixel centers -- they are the left edges of the pixels!'
    'digitize returns indices such that boxx[i-1] <= ptclx < boxx[i] .. but need to subtract 1 otherwise we have too much shifting'
    ptclx_indices = np.digitize(ptclx,boxx[:,0,0])-1
    ptcly_indices = np.digitize(ptcly,boxy[0,:,0])-1
    ptclz_indices = np.digitize(ptclz,boxz[0,0,:])-1

    return ptclx_indices,ptcly_indices,ptclz_indices,rstell,maxrstell,ptcld2o,ptclx,ptcly,ptclz


@profile
def smoothit(subhalostars,databox,boxx,boxy,boxz,ptclx_indices,ptcly_indices,ptclz_indices,rstell,maxrstell,ptcld2o,pixelsizekpc=0.25,distance=10.,normkey='Masses',photind=None,projection=None):
    ''

    'Choose how to weight distribution'
    if normkey == 'Masses':
        weights = subhalostars[normkey]*(1e10)/h
    else:
        weights = get_scaled_flux(subhalostars['GFM_StellarPhotometrics'][:,photind],distance)

    'Loop over particles and distribute mass'
    if verbose:
        print_verbose_string('Looping over particles and spreading out stellar mass or flux... (N = '+str(len(ptclx_indices))+' particles)')

    'Pixel volume'
    pixvol = pixelsizekpc**3

    'Convert rmax to delta pixels (dp)'
    dps = np.array([int(i) for i in maxrstell/pixelsizekpc])

    'Calculate gaussian normalizations'
    norms = pixvol * weights * (2*np.pi * rstell**2)**(-3./2)

    start_time = datetime.datetime.now()
    for ind in xrange(len(ptclx_indices)):
        pxi,pyi,pzi,sig,dp,norm,wi,rmax = ptclx_indices[ind],ptcly_indices[ind],ptclz_indices[ind],rstell[ind],dps[ind],norms[ind],weights[ind],maxrstell[ind]

        'Determine min/max indices (ie watch out for edges!)'
        xii,xif,yii,yif,zii,zif = get_mini_indices(databox.shape,pxi,pyi,pzi,dp)

        'Grab minibox and compute distances from the particle'
        miniboxdist = numexpr.evaluate('sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)',local_dict={"x":boxx[xii:xif, yii:yif, zii:zif],"y":boxy[xii:xif, yii:yif, zii:zif],"z":boxz[xii:xif, yii:yif, zii:zif],"x0":boxx[pxi,pyi,pzi],"y0":boxy[pxi,pyi,pzi],"z0":boxz[pxi,pyi,pzi]})
        
        'Fill in the minibox'
        databox[xii:xif, yii:yif, zii:zif] += numexpr.evaluate('A*exp(-0.5*(dist/s)**2)',local_dict={"A":norm,"dist":miniboxdist,"s":sig})

        del miniboxdist,pxi,pyi,pzi,sig,dp,norm,wi,rmax,xii,xif,yii,yif,zii,zif

    gc.collect()

    end_time = datetime.datetime.now()
    dt = end_time - start_time
    print ' .. Time [minutes] spent in loop: ',str(np.around(dt.seconds/60.,2))
    if normkey == 'Masses':
        print 'sum of databox:',np.sum(databox)
        print 'sum of stellar mass [Msun]:',np.sum(weights)
        print 'nbytes: ',databox.nbytes
    else:
        print 'sum of databox:',np.sum(databox)
        print 'sum of flux [1e24 erg/s/cm^2/Hz]:',np.sum(weights)
        print 'nbytes: ',databox.nbytes
    print ''

    del boxx,boxy,boxz,ptclx_indices,ptcly_indices,ptclz_indices,rstell,maxrstell,ptcld2o,dps,weights,norms
    gc.collect()

    return databox


def plot_smoothing_extent(maxrstell,subhalo_tag,pixelsize):
    f = plt.figure()
    ax = f.add_subplot(111)
    seaborn.distplot(maxrstell,kde=False,rug=False,color=pylab.cm.bone(0.5),bins=500,hist_kws=dict(edgecolor='none'))
    ax.axvline(pixelsize,lw=1,color=pylab.cm.YlOrRd(0.6))
    f.savefig(plotsdir+'/kneighbor/smooth_extent_dist_'+subhalo_tag+'.pdf')
    return None

def plot_kneighbors(ptclkdt,ptclcoords,subhalo_tag,pixelsize):
    'Calculate nearest neighbors for a few values of k'
    ptcldistances15,ids15 = ptclkdt.query(ptclcoords,k=15)
    ptcldistances10,ids10 = ptclkdt.query(ptclcoords,k=10)
    ptcldistances5,ids5 = ptclkdt.query(ptclcoords,k=5)
    ptcldistances3,ids3 = ptclkdt.query(ptclcoords,k=3)
    ptcldistances2,ids2 = ptclkdt.query(ptclcoords,k=2)

    use_cmap = pylab.cm.get_cmap('YlGnBu')
    YlOrRd = pylab.cm.get_cmap('YlOrRd')
    sp = (5./8)
    f = plt.figure()
    ax = f.add_subplot(111)
    seaborn.distplot(ptcldistances15[:,-1],kde=False,rug=False,color=use_cmap(0.9),bins=500,hist_kws=dict(edgecolor='none'))
    ax.text(60,sp*8e3,'n = 15',fontsize=15,fontweight='bold',color=use_cmap(0.9))

    seaborn.distplot(ptcldistances10[:,-1],kde=False,rug=False,color=use_cmap(0.7),bins=500,hist_kws=dict(edgecolor='none'))
    ax.text(60,(sp**2)*8e3,'n = 10',fontsize=15,fontweight='bold',color=use_cmap(0.7))

    seaborn.distplot(ptcldistances5[:,-1],kde=False,rug=False,color=use_cmap(0.5),bins=500,hist_kws=dict(edgecolor='none'))
    ax.text(60,(sp**3)*8e3,'n = 5',fontsize=15,fontweight='bold',color=use_cmap(0.5))

    seaborn.distplot(ptcldistances3[:,-1],kde=False,rug=False,color=use_cmap(0.3),bins=500,hist_kws=dict(edgecolor='none'))
    ax.text(60,(sp**4)*8e3,'n = 3',fontsize=15,fontweight='bold',color=use_cmap(0.3))

    seaborn.distplot(ptcldistances2[:,-1],kde=False,rug=False,color=use_cmap(0.1),bins=500,hist_kws=dict(edgecolor='none'))
    ax.text(60,(sp**5)*8e3,'n = 2',fontsize=15,fontweight='bold',color=use_cmap(0.1))

    ax.set_yscale('log')
    ax.set_xlim(0,100)
    ax.set_xlabel('Distance to kth nearest neighbor [kpc]')
    ax.set_ylabel('N particles')
    ax.set_title('Adaptive smoothing: Distribution of kernel sizes')

    'denote pixel size'
    ax.axvline(pixelsize,lw=1,color=YlOrRd(0.6))

    f.savefig(plotsdir+'/kneighbor/nearestnthneighbor_dist_'+subhalo_tag+'.pdf')
    return None


@profile
def run_single_subhalo(subhaloID):
    if verbose:
        print_verbose_string('Loading particle data for Subhalo '+str(subhaloID)+'...')

    'Make sure there are actually stars here, and get rid of any wind particles'
    subhalostarsandwinds = il.snapshot.loadSubhalo(TNGdir+'/output/',snapnum,subhaloID,'stars')
    check_for_stars(subhalostarsandwinds)
    subhalostars = exclude_wind_particles(subhalostarsandwinds)

    'Check to make sure particles associated with this subhalo are not wrapped around the other side of the periodic box'
    subhalostars = check_edge_effects(subhalostars)

    'Make any adjustments, if needed'
    subhalostars = make_adjustments(subhalostars,adjustment=adjustment)

    'Rotation?'
    #subhalostars = tt.calc_angmom.rotate_galaxy(subhalostars,Rout=10,direction=projection,cookey='CoordinatesShifted')
    subhalostars = tt.calc_angmom.rotate_galaxy(subhalostars,Rout=10,direction=projection,cookey='CoordinatesAdjusted')
    
    'Create placeholder empty box'
    emptybox0,boxx,boxy,boxz = build_box(subhalostars,pixelsizekpc)

    'Determine particle indices and pp distances etc'
    ptclx_indices,ptcly_indices,ptclz_indices,rstell,maxrstell,ptcld2o,ptclx,ptcly,ptclz = measure_particleparticle(subhalostars,emptybox0,boxx,boxy,boxz,
                                                                                                                    kneighbors=3,pixelsizekpc=pixelsizekpc,distance=Mpc,projection=projection,
                                                                                                                    outtag='subhalo'+str(specialSubhalo)+'_'+tag_base)

    'Now fill in the box with either mass or light distributions'
    for normkey,phind,oname in zip(boxtypes['normkey'],boxtypes['photind'],boxtypes['name']):
        'Define subhalo tag'
        subhalo_tag = 'subhalo'+str(specialSubhalo)+'_'+tag_base+'_'+oname

        'Set output directory'
        if normkey == 'Masses':
            outdir = boxdir +'/'+myTNG+ '/stellarmass/allstars/'
        else:
            outdir = boxdir +'/'+myTNG+ '/light/allstars/'

        mkdirp(outdir)

        if not overwrite:
            if os.path.isfile(outdir+'/'+subhalo_tag+'.hdf5'):
                print_verbose_string('File exists already (and we are not in overwrite mode)! Skipping! ')
                continue

        if verbose:
            print_verbose_string('Working on '+oname+'...')

        'Create 2D histogram of particles ...'
        if verbose:
            print_verbose_string('Creating 2D histogram (xy) ...')

        make_hist_2d(subhalostars,subhalo_tag,normkey=normkey,photind=phind,distance=Mpc,xbins=boxx[:,0,0],ybins=boxy[0,:,0],outdir=outdir)

        'Make sure this is really zeros! (had some issues with this before)'
        emptybox = np.zeros(emptybox0.shape)

        'Smooth mass or light around in the box..!'
        databox = smoothit(subhalostars,emptybox,boxx,boxy,boxz,ptclx_indices,ptcly_indices,ptclz_indices,rstell,maxrstell,ptcld2o,
                           pixelsizekpc=pixelsizekpc,distance=Mpc,normkey=normkey,photind=phind,projection=projection)

        'Array is too big. Decrease precision.'
        #databox2 = databox.astype(np.float32)
        #print databox.nbytes
        #print databox2.nbytes

        'Save all relevant outputs.'
        save_outputs(subhaloID,subhalo_tag,outdir,data=databox,sig=rstell,maxr=maxrstell,x=ptclx,y=ptcly,z=ptclz,disttype=oname,dustmodel='None',
                     dist=Mpc,pixkpc=pixelsizekpc,proj=projection,littleh=h,snapnum=snapnum)
        #save_outputs(subhaloID,subhalo_tag,outdir,data=databox2,sig=rstell,maxr=maxrstell,x=ptclx,y=ptcly,z=ptclz,disttype=oname,dustmodel='None',
        #             dist=Mpc,pixkpc=pixelsizekpc,proj=projection,littleh=h,snapnum=snapnum)


        'Clean up / free memory for this particular box'
        #del emptybox,databox,databox2
        del emptybox,databox
        gc.collect()

    'Clean up / free memory for this subhalo'
    del emptybox0,boxx,boxy,boxz,ptclx_indices,ptcly_indices,ptclz_indices,rstell,maxrstell,ptcld2o,ptclx,ptcly,ptclz
    gc.collect()
    
    print '\nDone with Subhalo'+str(subhaloID)+'...\n'

    return None

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    check_input(arguments)

    'Mode options'
    verbose = arguments['--verbose']
    overwrite = arguments['--overwrite']

    'Directory structure'
    TNGdir = arguments['--TNGdir']
    adjustdir = arguments['--adjustdir']
    plotsdir = arguments['--plotsdir']
    mkdirp(plotsdir)
    boxdir = arguments['--boxdir']
    mkdirp(boxdir)
    
    'Simulation details'
    snapnum = int(arguments['--snapnum'])
    h = np.float(arguments['--littleh'])
    a = grab_scalefactor(snapnum)

    myTNG = TNGdir.split('/')[-2]
    check_TNG()



    'Output box details'
    sdss_g = arguments['--gband']
    sdss_r = arguments['--rband']
    sdss_i = arguments['--iband']
    mass = arguments['--mass']
    boxtypes = consolidate_output_choices()

    projection = arguments['--projection']
    pixelsizekpc = np.float(arguments['--pixelsize'])
    kneighbors = int(arguments['--kernelsize'])
    maxNkernels = int(arguments['--multkernel'])
    Mpc = np.float(arguments['--distance'])

    'Making any ajustments?'
    adjustment = arguments['--adjustment']

    'Input Subhalo ID(s)'
    specialSubhalo = arguments['--subhaloID']
    subhalofile = arguments['--filename']

    tag_base = 'proj_'+projection+'_Mpc_'+str(Mpc)+'_k_'+str(kneighbors)+'_kpcpix_'+str(pixelsizekpc)+'_adjust_'+str(adjustment)

    if verbose:
        print '\nRunning with settings:\n',arguments,'\n'

    if specialSubhalo:
        'Load the stellar particles for this subhalo'
        subhaloID = int(specialSubhalo)

        t0 = datetime.datetime.now()

        'Run this subalo'
        run_single_subhalo(subhaloID)

        t1 = datetime.datetime.now()
        
        print 'Running subhalo'+str(subhaloID)+' took '+str(np.around((t1-t0).total_seconds()/60.,2))+' minutes'

        print '\nDone.\n'



    else:
        'Load the stellar particles for the entire snapshot'
        snapstars = load_snapshot_or_whatever

        for subhaloID in subhaloIDs:
            'Run this subhalo'
            run_single_subhalo(subhaloID)

        print '\nDone.\n'
