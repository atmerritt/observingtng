#!/usr/bin/env python
""" butterfly.py -- Given a 3D grid of the stellar mass or light distribution for a TNG subhalo, create realistic-ish "Dragonfly" images. Note: if stellar mass, all this does is put things in dragonfly-sized pixels (for ease of comparison w images); ie "-w" is False at all times for mass.

Usage: 
    butterfly.py [-h] [-v] [-O] [-B] [-g] [-r] [-m] [-w] [--spatres ARCSEC] [--refimage STR] [--axcollapse STR] [--projection TYPE] [--distance MPC] [--kpcperpix KPC] [--kernelsize INT] [--myTNG TNG] [--adjustment TYPE] [--snapnum INT] [--plotsdir DIR] [--boxdir DIR] [--imagedir DIR] [--psfdir DIR] [--backdir DIR] <subfindID>

Options:
    -h, --help                    Show this screen                                                                          [default: False]
    -v, --verbose                 Print extra information                                                                   [default: False]
    -O, --overwrite               Overwrite existing frames                                                                 [default: False]
    -B, --hist2d                  No binning (use hist2d)                                                                   [default: False]

    -g, --gband                   Create g-band image (SDSS filters)                                                        [default: False]
    -r, --rband                   Create r-band image (SDSS filters)                                                        [default: False]
    -m, --mass                    Create stellar mass image                                                                 [default: False]
    -w, --works                   "The works" .. ie make it look like dragonfly (else: idealized)                           [default: False]

    -a ARCSEC, --spatres ARCSEC   Spatial resolution in arcsec per pixel                                                    [default: 2.5]
    -z STR, --refimage STR        Name of DNGS reference image to use                                                       [default: M101]

    -y STR, --axcollapse STR      Choice of axis to collapse databox along ('x','y', or 'z')                                [default: z]
    -p TYPE, --projection TYPE    Optionally specify a Faceon or Edgeon image                                               [default: None]
    -d MPC, --distance MPC        Distance to place galaxy at (in Mpc)                                                      [default: 10.]
    -x KPC, --kpcperpix KPC       Kpc per pix in 3D box                                                                     [default: 0.5]
    -k INT, --kernelsize INT      Smooth particles with kernel sizes corresponding to the kth nearest neighbor              [default: 3]

    -T TNG, --myTNG TNG           Version of TNG                                                                            [default: L75n1820TNG]
    -j TYPE, --adjustment TYPE    "exsitu","insitucontract", "postinfall", "demigrate", "mergers-[123]" or "mergers-4-z[1,2,20]-f0[57]"      [default: None]
    -s INT, --snapnum INT         Snap number                                                                               [default: 99]

    -q DIR, --psfdir DIR          Directory where Dragonfly PSF file lives                                                  [default: /u/allim/Projects/TNG/DNGS_0.9/PSF/]
    -b DIR, --backdir DIR         Directory where Dragonfly background frames live                                          [default: /u/allim/Projects/TNG/DNGS_0.9/]
    -t DIR, --plotsdir DIR        Directory where diagnostic plots will live                                                [default: /u/allim/Projects/TNG/Plots/butterfly/]
    -o DIR, --imagedir DIR        Directory where output 2D FITS frames will live                                           [default: /u/allim/Projects/TNG/FITS/]
    -c DIR, --boxdir DIR          Directory where 3D grids with mass or light distributions live                            [default: /u/allim/Projects/TNG/Boxes/]

Example: 
    python butterfly.py -v -g -r  505333 (light images, idealized)
    python butterfly.py -v -g -r -w 505333 (light images, dragonfly-ified)
    python butterfly.py -v -m 505333 (stellar mass images)

"""

from __future__ import division

import sys
import os

if not os.path.isfile('login.cl'):
    print '\n********************************************************'
    print ' To run this script, first create a login.cl file via:'
    print ' $ mkiraf'
    print '********************************************************\n'
    sys.exit()


try:
    from pyraf import iraf
except ImportError:
    print '\n********************************************************'
    print ' To run this script, enter the irafenv environment via:'
    print ' $ iraf_on'
    print '********************************************************\n'
    sys.exit()


import matplotlib as ml
ml.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc,colorbar,lines

from scipy import signal

from astropy.io import fits,ascii
from astropy.table import Table
from astropy.visualization import ImageNormalize,ZScaleInterval,LogStretch,LinearStretch,MinMaxInterval
import astropy.units as u

import tng_tools as tt
import numpy as np
import seaborn
import docopt
import pylab
import h5py
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

def check_overwrite(fname):
    if os.path.isfile(fname):
        if not overwrite:
            print '*** File exists already, and we are not in overwrite mode! Skipping. ***\n'
            sys.exit()
    return None

def load_iraf_packages():
    if verbose:
        print_verbose_string('Loading IRAF packages ...')

    iraf.xray()
    iraf.xray.xspatial()
    return None

def get_zeropoints(refname):
    if refname == 'NGC1042':
        ZPg,ZPr = 16.2,16.08

    elif refname == 'M101':
        ZPg,ZPr = 19.95,19.02
    return ZPg,ZPr


def write_FITS(data,saveas,head_dict=None):
    hdu = fits.PrimaryHDU(data=data)
    
    if head_dict is not None:
        for keyname in head_dict.keys():
            hdu.header[keyname] = head_dict[keyname]

    hdulist = fits.HDUList([hdu])
    hdulist.writeto(saveas,overwrite=True)
    hdulist.close()

    return None

def clean_up():
    if verbose:
        print_verbose_string('Cleaning up ...')
    for fname in ['_im.fits','_im_at_distance.fits','_im_counts.fits','_im_smoothed.fits']:
        clearit(fname)
    return None

def check_for_box(boxname):
    box_exists = os.path.isfile(boxname)
    return box_exists


def make_butterfly_image(im_counts,refback):
    'smooth image with PSF'
    'CROP pre-determined area (save in dictionary or something) of reference frame and add that to image (ie do *not* place image in a blank area of ref frame! want to be able to use the same isophotes)'
    'Save images:'
    '--> + reference frame section, star-subtracted'

    myPSFname = PSFdir+'/M101_psf_good.fits'.replace('//','/')    
    psf = fits.getdata(myPSFname)
    
    if verbose:
        print_verbose_string('Convolving idealized frame with the Dragonfly PSF ...')
    smoothed = signal.convolve2d(im_counts,psf,mode='same')
    write_FITS(smoothed,'_im_smoothed.fits')


    if verbose:
        print_verbose_string('Adding in a realistic DNGS background...')

    'grab a 3000x3000 pixel region from the reference image'
    ref_rc,ref_cc = 1935,1535
    halfsize = 1500
    refbox = refback[ref_rc-halfsize:ref_rc+halfsize,ref_cc-halfsize:ref_cc+halfsize]

    'stick that in the middle of the smoothed TNG galaxy image'
    bfly = np.array(smoothed)

    xpix,ypix = np.meshgrid(np.arange(bfly.shape[1]),np.arange(bfly.shape[0]))
    gal_cc = int(xpix.ravel()[bfly.ravel()==np.max(bfly.ravel())][0])
    gal_rc = int(ypix.ravel()[bfly.ravel()==np.max(bfly.ravel())][0])

    if bfly.shape[0] <= refbox.shape[0] and bfly.shape[1] <= refbox.shape[1]:
        'if the galaxy image is smaller than the background, then just place it in the corner'
        print 'image fits in refbox!'
        bfly += refbox[:bfly.shape[0],:bfly.shape[1]]

    else: 
        'if the galaxy image is larger along either dimension... '
        print 'need to make some adjustments!'
        print 'image:',bfly.shape
        print 'refbox:',refbox.shape
        brow_i,brow_f = 0,bfly.shape[0]
        bcol_i,bcol_f = 0,bfly.shape[1]
        if bfly.shape[0] > refbox.shape[0]:
            'update rows'
            brow_i = np.max((0,gal_rc-halfsize))
            brow_f = np.min((bfly.shape[0],gal_rc+halfsize))
            print brow_i,brow_f,'-->',brow_f-brow_i

        if bfly.shape[1] > refbox.shape[1]:
            'update columns'
            bcol_i = np.max((0,gal_cc-halfsize))
            bcol_f = np.min((bfly.shape[1],gal_cc+halfsize))
            print bcol_i,bcol_f,'-->',bcol_f-bcol_i

        refrow_i,refrow_f = 0,brow_f-brow_i
        refcol_i,refcol_f = 0,bcol_f-bcol_i

        bfly[brow_i:brow_f,bcol_i:bcol_f] += refbox[refrow_i:refrow_f,refcol_i:refcol_f]

    return bfly

def make_single_image(boxname,outname,mass_only=False,g_only=False,r_only=False):
    '(0) Read in box data from hdf5 file; with a particular dimension specified'
    h5 = h5py.File(boxname,'r')
    if hist2d:
        im = h5.get('h2d').value
    else:
        im = h5.get('box_'+xyz).value

    '(1) Write that out to a temporary file'
    write_FITS(im,'./_im.fits')


    '(2) Put 2D image into Dragonfly-sized pixels'
    '... first figure out how many kpc Dragonfly pixels are at this distance'
    radperpix = (arcsecperpix*u.arcsec).to(u.radian).value
    dist_kpc = (distanceMpc*u.Mpc).to(u.kpc).value
    dragonfly_kpcperpix = dist_kpc*radperpix

    '... then use iraf to change the pixel scale for the TNG frame'
    pixratio = kpcperpix/dragonfly_kpcperpix
    iraf.magnify(input='_im.fits',output='_im_at_distance.fits',xmag=pixratio,ymag=pixratio)

    '... and read in image at distance'
    im_distpix = fits.getdata('_im_at_distance.fits')

    if mass_only:
        'Ok we are done. Save output FITS file '
        write_FITS(im_distpix,outname)



    '(3) If light: scale flux appropriately; if --theworks, do a bunch of other stuff too'
    ' ... note this is where we undo the 1e24 scaling used in adaptivebox'
    ZPg,ZPr = get_zeropoints(refname)

    if g_only:
        im_counts = 10**( ( -2.5*np.log10(im_distpix/1e24) - 48.6 - ZPg )/(-2.5) )

        if theworks:
            'Do the works'
            im_obs = make_butterfly_image(im_counts,refim_g)
            write_FITS(im_obs,outname,head_dict={'band':'gband','ZP':ZPg,'ref':refname})

        else:
            'Save output FITS file'
            write_FITS(im_counts,outname,head_dict={'band':'gband','ZP':ZPg,'ref':refname})


    if r_only:
        im_counts = 10**( ( -2.5*np.log10(im_distpix/1e24) - 48.6 - ZPr )/(-2.5) )

        if theworks:
            'Do the works'
            im_obs = make_butterfly_image(im_counts,refim_r)
            write_FITS(im_obs,outname,head_dict={'band':'rband','ZP':ZPr,'ref':refname})

        else:
            'Save output FITS file'
            write_FITS(im_counts,outname,head_dict={'band':'rband','ZP':ZPr,'ref':refname})


    'Clean up'
    clean_up()

    if verbose:
        print_verbose_string('FITS file saved under: '+outname)
        print ''

    return None


if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)

    'Mode options'
    verbose = arguments['--verbose']
    overwrite = arguments['--overwrite']

    'Directory structure'
    boxdir = arguments['--boxdir']
    FITSdir = arguments['--imagedir']
    plotsdir = arguments['--plotsdir']
    PSFdir = arguments['--psfdir']
    backdir = arguments['--backdir']

    myTNG = arguments['--myTNG']
    boxdir = boxdir + '/'+ myTNG
    snapnum = int(arguments['--snapnum'])

    'Output image details'
    sdss_g = arguments['--gband']
    sdss_r = arguments['--rband']
    mass = arguments['--mass']

    theworks = arguments['--works']
    refname = arguments['--refimage']
    hist2d = arguments['--hist2d']

    arcsecperpix = np.float(arguments['--spatres'])
    distanceMpc = np.float(arguments['--distance'])
    kpcperpix = np.float(arguments['--kpcperpix'])
    kernelsize = int(arguments['--kernelsize'])

    adjustment = arguments['--adjustment']

    projection = arguments['--projection']
    xyz = 'xyz'.replace(arguments['--axcollapse'],'')

    'Subhalo in question'
    subfindID = int(arguments['<subfindID>'])

    if verbose:
        print '\n',arguments,'\n'

    'Ready goooo'
    if mass:
        boxname1 = boxdir + '/stellarmass/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_'+str(kernelsize)+'_kpcpix_'+str(kpcperpix)+'_adjust_'+adjustment+'_StellarMass_snap'+str(snapnum)+'.hdf5'
        boxname2 = boxdir + '/stellarmass/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_'+str(kernelsize)+'_kpcpix_'+str(kpcperpix)+'_StellarMass_extra.hdf5'

        print boxname1
        print boxname2
        box_exists = check_for_box(boxname1)

        if box_exists:
            boxname = boxname1
        else:
            box_exists = check_for_box(boxname2)
            if box_exists:
                boxname = boxname2

        if box_exists:

            if hist2d:
                boxname = boxname.replace('.hdf5','_hist2d.hdf5')
                kernelsize = 0

            if verbose:
                print_verbose_string('Found file: '+boxname)
                print_verbose_string('Creating stellar mass image for SubfindID '+str(subfindID)+'...')

            outname = FITSdir + '/stellarmass/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_'+str(kernelsize)+'_kpcpix3d_'+str(kpcperpix)+'_adjust_'+adjustment+'_snap'+str(snapnum)+'_arcsec_'+str(arcsecperpix)+'_'+xyz+'_mstell.fits'

            check_overwrite(outname)
            make_single_image(boxname,outname,mass_only=True)
        else:
            print '** File does not exist, skipping: \n'+str(subfindID)+'\n'

    'If necessary, load IRAF packages for PSF convolution'
    if theworks:
        load_iraf_packages()
        refim_g = fits.getdata(backdir+'/'+refname+'/'+refname+'_diff_g.fits')
        refim_r = fits.getdata(backdir+'/'+refname+'/'+refname+'_diff_r.fits')


    if sdss_g:
        boxname1 = boxdir + '/light/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_3_kpcpix_'+str(kpcperpix)+'_adjust_'+adjustment+'_gband_snap'+str(snapnum)+'.hdf5'
        boxname2 = boxdir + '/light/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_3_kpcpix_'+str(kpcperpix)+'_gband_extra.hdf5'
        box_exists = check_for_box(boxname1)

        if box_exists:
            boxname = boxname1
        else:
            box_exists = check_for_box(boxname2)
            if box_exists:
                boxname = boxname2

        if box_exists:

            if theworks:
                outname = FITSdir + '/light/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_'+str(kernelsize)+'_kpcpix3d_'+str(kpcperpix)+'_adjust_'+adjustment+'_snap'+str(snapnum)+'_arcsec_'+str(arcsecperpix)+'_'+xyz+'_gband_obs.fits'
                check_overwrite(outname)
                if verbose:
                    print_verbose_string('Creating gband light image for SubfindID '+str(subfindID)+', complete with observational effects...')
            else:
                outname = FITSdir + '/light/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_'+str(kernelsize)+'_kpcpix3d_'+str(kpcperpix)+'_adjust_'+adjustment+'_snap'+str(snapnum)+'_arcsec_'+str(arcsecperpix)+'_'+xyz+'_gband_ideal.fits'
                check_overwrite(outname)
                if verbose:
                    print_verbose_string('Creating gband light image for SubfindID '+str(subfindID)+', idealized version...')

            make_single_image(boxname,outname,g_only=True)
        else:
            print '** File does not exist, skipping: \n'+boxname+'\n'



    if sdss_r:
        boxname1 = boxdir + '/light/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_3_kpcpix_'+str(kpcperpix)+'_adjust_'+adjustment+'_rband_snap'+str(snapnum)+'.hdf5'
        boxname2 = boxdir + '/light/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_3_kpcpix_'+str(kpcperpix)+'_rband_extra.hdf5'
        box_exists = check_for_box(boxname1)

        if box_exists:
            boxname = boxname1
        else:
            box_exists = check_for_box(boxname2)
            if box_exists:
                boxname = boxname2

        if box_exists:
            if theworks:
                outname = FITSdir + '/light/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_'+str(kernelsize)+'_kpcpix3d_'+str(kpcperpix)+'_adjust_'+adjustment+'_snap'+str(snapnum)+'_arcsec_'+str(arcsecperpix)+'_'+xyz+'_rband_obs.fits'
                check_overwrite(outname)
                if verbose:
                    print_verbose_string('Creating rband light image for SubfindID '+str(subfindID)+', complete with observational effects...')
            else:
                outname = FITSdir + '/light/allstars/subhalo'+str(subfindID)+'_proj_'+projection+'_Mpc_'+str(distanceMpc)+'_k_'+str(kernelsize)+'_kpcpix3d_'+str(kpcperpix)+'_adjust_'+adjustment+'_snap'+str(snapnum)+'_arcsec_'+str(arcsecperpix)+'_'+xyz+'_rband_ideal.fits'
                check_overwrite(outname)
                if verbose:
                    print_verbose_string('Creating rband light image for SubfindID '+str(subfindID)+', idealized version...')

            make_single_image(boxname,outname,r_only=True)
        else:
            print '** File does not exist, skipping: \n'+boxname+'\n'

    

    print '\nDone with SubfindID '+str(subfindID)+' \n\n'
