#!usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from glob import glob
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from DISSK import utilities
from PyAstronomy.pyasl import helcorr

###########################################################################
"""Import all the necessary files and initialize directories"""
###########################################################################

homedir = '/uufs/astro.utah.edu/common/home/u1267339/'
visit_dir = '/uufs/chpc.utah.edu/common/home/sdss/dr17/apogee/spectro/redux/dr17/visit/'
stars_dir = '/uufs/chpc.utah.edu/common/home/sdss/dr17/apogee/spectro/redux/dr17/stars/'

apogee_wave = np.loadtxt(homedir + 'InputData/lambdatemp.txt')

allVisit = Table.read('/uufs/chpc.utah.edu/common/home/sdss/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allVisit-dr17-synspec_rev1.fits', format='fits')
allVisit = allVisit['APOGEE_ID', 'RA', 'DEC', 'VHELIO', 'BC', 'FILE', 'FIBERID', 'PLATE', 'MJD', 'FIELD', 'TELESCOPE']

apo_latitude = 32.78036
apo_longitude = -105.82042
apo_altitude = 2788 # meters


###########################################################################
###########################################################################
###########################################################################
"""This function gathers all the information from the apStar files and prepares the spectra to stack"""
###########################################################################

def prep_chip_to_stack(apStar, chip, bin_width=1000, jackknife_visit = None, bary_corr = True, spec_type='galaxy'):
    """ IDENTIFY REGIONS WHERE ALL VISITS HAVE DATA """
    start, stop = int(np.max(apStar['chips'][:,2*chip])), int(np.min(apStar['chips'][:,2*chip+1]))
    chip_apStar = dict(wave = [], flux = [], error = [], mask = [], sky = [], cont = [], snrs = [], rvs = apStar['rvs'], lim = np.array([start, stop]))


    for visit in range(apStar['nvis']):
        """ UNSHIFT SPECTRA BY BARYCENTRIC CORRECTIONS OR HELIOCENTRIC VELOCITIES, DEPENDING """

        chip_apStar['wave'].append(apogee_wave[start:stop])
        chip_apStar['flux'].append(utilities.unshift(apStar['flux'][visit][start:stop], apogee_wave[start:stop], 
                                           apStar['bcs'][visit]))
        chip_apStar['error'].append(utilities.unshift(apStar['error'][visit][start:stop], apogee_wave[start:stop], 
                                            apStar['bcs'][visit]))
        chip_apStar['sky'].append(utilities.unshift(apStar['sky'][visit][start:stop], apogee_wave[start:stop], 
                                          apStar['bcs'][visit]))

        """ IDENTIFY ALL MASKED PIXELS AND CALCULATE CONTINUUM """
        skyline_mask, chip_cont = utilities.identify_skylines_and_continuum(chip_apStar['wave'][visit], chip_apStar['flux'][visit], chip_apStar['sky'][visit], apStar['fibers'][visit], bin_width, spec_type=spec_type)
        
        chip_mask = ~(~apStar['mask'][visit][start:stop].astype(bool) + ~skyline_mask.astype(bool))

        chip_apStar['mask'].append(chip_mask)
        chip_apStar['cont'].append(chip_cont)
        
        chip_apStar['snrs'].append(np.nanmedian(chip_apStar['flux'][visit][chip_apStar['mask'][visit]]/
                                          chip_apStar['error'][visit][chip_apStar['mask'][visit]]))

    # Convert entries to numpy arrays (with mask conversion to bool)
    for key in ['wave', 'flux', 'error', 'sky', 'mask', 'cont', 'snrs', 'rvs']:
        chip_apStar[key] = np.array(chip_apStar[key], dtype=bool if key == 'mask' else float)
    
    chip_apStar['nvis'] = len(chip_apStar['snrs'])
    
    # Jackknife removal of specific visits
    if jackknife_visit is not None:
        keep_inds = np.ones(chip_apStar['nvis'], dtype=bool)
        keep_inds[jackknife_visit] = False
        for key in ['wave', 'flux', 'error', 'sky', 'mask', 'cont', 'snrs', 'rvs']:
            chip_apStar[key] = chip_apStar[key][keep_inds]
        chip_apStar['nvis'] -= 1
    
    # Remove visits with low SNR
    vis_snr_cutoff = 1.0
    if len(chip_apStar['snrs']) > 1:
        good_visits = chip_apStar['snrs'] > vis_snr_cutoff
        if np.any(~good_visits):
            print(f'Removing {np.count_nonzero(~good_visits)} Low-SNR Visits')
        
        for key in ['wave', 'flux', 'error', 'sky', 'mask', 'cont', 'rvs']:
            chip_apStar[key] = chip_apStar[key][good_visits]
        
        chip_apStar['nvis'] = len(chip_apStar['snrs'][good_visits])


    return chip_apStar


###########################################################################
###########################################################################
###########################################################################
"""This function stacks all the visits together"""
###########################################################################

def stack_visits(chip_apStar, return_as_dict=False):

    """ INITIALIZE RETURN DICTIONARY """

    if chip_apStar['wave'].ndim > 1:
        wave = chip_apStar['wave'][0]
    else:
        wave = chip_apStar['wave']



    
    shape = np.shape(wave)
    return_chip = dict(wave = wave, flux = np.zeros(shape), error = np.zeros(shape), mask = np.ones(shape, dtype=bool), sky = np.zeros(shape), cont = np.zeros(shape), rvs = np.nanmean(chip_apStar['rvs']))

    """ CONTINUUM NORMALIZE FLUX AND ERROR ARRAYS """
    chip_apStar['flux'] = chip_apStar['flux']/chip_apStar['cont']
    chip_apStar['error'] = chip_apStar['error']/chip_apStar['cont']


    wave_inds = np.where((apogee_wave >= return_chip['wave'][0]) & (apogee_wave <= return_chip['wave'][-1]))[0]
    return_chip['lim'] = np.array([wave_inds[0], wave_inds[-1]+1], dtype=int)

    for pixel in range(len(return_chip['wave'])):

        """ SKY FLUX AND CONTINUUM ARE AVERAGED OVER ALL VISITS REGARDLESS OF MASKS """
        return_chip['sky'][pixel] = np.nanmean(chip_apStar['sky'][:,pixel])
        return_chip['cont'][pixel] = np.nanmean(chip_apStar['cont'][:,pixel])

        """ IF A PIXEL IS MASKED IN AT LEAST HALF OF THE VISITS, IT'S MASKED IN THE STACKED SPECTRUM,
            BUT THE FLUX AND ERROR ARRAYS ARE CALCULATED SANS MASKS """
        if len(np.where(chip_apStar['mask'][:,pixel] == False)[0]) > chip_apStar['nvis']/2:

            numerator = np.sum(chip_apStar['flux'][:,pixel]/
                               chip_apStar['error'][:,pixel]**2)
            denominator = np.sum(1/chip_apStar['error'][:,pixel]**2)

            return_chip['flux'][pixel] = numerator / denominator
            return_chip['error'][pixel] = 1 / denominator

            return_chip['mask'][pixel] = False

        else:
            """ OTHERWISE, THE FLUX AND ERROR IN THAT PIXEL IS CALCULATED 
            EXCLUDING PIXELS MASKED IN THE VISITS """
            
            numerator = np.sum(chip_apStar['flux'][chip_apStar['mask'][:,pixel],pixel]/
                               chip_apStar['error'][chip_apStar['mask'][:,pixel],pixel]**2)
            denominator = np.sum(1/chip_apStar['error'][chip_apStar['mask'][:,pixel],pixel]**2)
    
            return_chip['flux'][pixel] = numerator / denominator
            return_chip['error'][pixel] = 1 / denominator

        if np.isnan(return_chip['flux'][pixel]):
            return_chip['mask'][pixel] = False

    # if return_as_dict:
    return return_chip
    # else:
    #     return np.vstack([return_chip['wave'], return_chip['flux'], return_chip['error'], return_chip['mask'].astype(bool), return_chip['cont'], return_chip['sky'], return_chip['lim']])


###########################################################################
###########################################################################
###########################################################################
"""These two functions will locate and return all the apVisit or apStar files in dr17 corresponding to an APOGEE ID"""
###########################################################################

def locate_apStar(apogee_id):

    these_visits = allVisit[allVisit['APOGEE_ID'] == apogee_id]

    filenames = []

    for visit in range(len(these_visits)):

        filenames.append(glob(stars_dir + str(these_visits['TELESCOPE'][visit]) + '/' + str(these_visits['FIELD'][visit]) + '/a*Star-dr17-' + apogee_id + '.fits')[0])

    return filenames

def locate_apVisit(apogee_id):

    these_visits = allVisit[allVisit['APOGEE_ID'] == apogee_id]

    filenames = []

    for visit in range(len(these_visits)):

        filenames.append(visit_dir + str(these_visits['TELESCOPE'][visit]) + '/' + str(these_visits['FIELD'][visit]) + '/' + str(int(these_visits['PLATE'][visit])) + '/' + str(these_visits['MJD'][visit]) + '/' + str(these_visits['FILE'][visit]))

    return filenames



###########################################################################
###########################################################################
###########################################################################
"""This function collects everything necessary from the apStar files and organizes it"""
###########################################################################

def collect_data_from_apStar(file, bary_corr=True):

    # apStar = dict(flux = np.array([]), error = np.array([]), mask = np.array([]), sky = np.array([]), fibers = np.array([]), rvs = np.array([]), 
    #               ras = 0, dec = 0, nvis = 0)
    apStar = dict()

    """READ IN FIBER SPECTRAL DATA, REMOVING THE FIRST TWO ENTRIES AS THEY ARE COMBINED."""
    hdulist = fits.open(file)
    apStar['ID'] = hdulist[0].header['OBJID']
    apStar['nvis'] = hdulist[0].header['NVISITS']
    apStar['ras'] = hdulist[0].header['RA']
    apStar['dec'] = hdulist[0].header['DEC']
    
    apStar['flux'] = hdulist[1].data[2:,:]
    apStar['flux'][np.isnan(apStar['flux'])] = 0.0
    apStar['error'] = hdulist[2].data[2:,:]
    bitmasks = hdulist[3].data[2:,:]
    apStar['mask'] = np.zeros(np.shape(bitmasks), dtype=bool)
    apStar['sky'] = hdulist[4].data[2:,:]

    apStar['fibers'] = np.array([])
    apStar['rvs'] = np.array([])
    apStar['chips'] = np.zeros((apStar['nvis'], 6))

    hdulist.close()

    """READ IN FIBER METADATA."""
    if bary_corr:
        rvname = 'BC'
    else:
        rvname = 'VHELIO'
    

    for visit in range(apStar['nvis']):
        
        rvname += str(visit+1)
        fibername = 'FIBER' + str(visit+1)
        
        apStar['rvs'] = np.append(apStar['rvs'], hdulist[0].header[rvname])
        apStar['fibers'] = np.append(apStar['fibers'], hdulist[0].header[fibername])


        ### EVALUATE PIXEL-LEVEL BITMASKS
        apStar['mask'][visit] = utilities.identify_bitmasks(bitmasks[visit]).astype(bool)

    if not bary_corr:
        apStar['rvs'] *= -1

    """SOMETIMES THERE ARE VISITS THAT HAVE NO DATA? REMOVE THEM"""
    if not np.all(np.all(apStar['flux'] == 0, axis=1) == False): # if there is at least one visit that is all zeros
        print('ARRR HERE THERE BE EMPTY COLUMNS')
        
        nonzero_cols = np.any(apStar['flux'] !=0, axis=1)
        apStar['flux'] = apStar['flux'][nonzero_cols]
        apStar['error'] = apStar['error'][nonzero_cols]
        apStar['sky'] = apStar['sky'][nonzero_cols]
        apStar['mask'] = apStar['mask'][nonzero_cols].astype(bool)
        apStar['fibers'] = apStar['fibers'][nonzero_cols]
        apStar['rvs'] = apStar['rvs'][nonzero_cols]
        apStar['nvis'] = len(apStar['flux'])
        
    for visit in range(apStar['nvis']):
        ### FIND WHERE THE CHIPS END
        apStar['chips'][visit] = utilities.identify_chip_ends(apStar['flux'][visit])
        
    return apStar


###########################################################################
###########################################################################
###########################################################################
"""This function collects everything necessary from the apVisit files and organizes it"""
###########################################################################

def collect_data_from_apVisit(apoid, fix_chip_gaps=False):

    these_visits = allVisit[allVisit['APOGEE_ID'] == apoid]

    apStar = dict(flux = [], error = [], mask = [], sky = [])
    apStar['ID'] = apoid
    apStar['nvis'] = len(these_visits)
    apStar['ras'] = np.average(these_visits['RA'])
    apStar['dec'] = np.average(these_visits['DEC'])

    apStar['fibers'] = np.array(these_visits['FIBERID'])
    apStar['chips'] = np.zeros((apStar['nvis'], 6))
    apStar['rvs'] = np.array(these_visits['VHELIO'])
    apStar['bcs'] = np.array([])
        

    for visit in range(len(these_visits)):

        filename = visit_dir + str(these_visits['TELESCOPE'][visit]) + '/' + str(these_visits['FIELD'][visit]) + '/' + str(int(these_visits['PLATE'][visit])) + '/' + str(these_visits['MJD'][visit]) + '/' + str(these_visits['FILE'][visit]) 
        hdulist = fits.open(filename)

        chip_flux, chip_error, chip_mask, chip_sky = [], [], [], []
        for chip in range(3):
            f = interp1d(hdulist[4].data[chip], hdulist[1].data[chip], bounds_error=False, fill_value=np.nan)
            chip_flux.append(f(apogee_wave))
            f = interp1d(hdulist[4].data[chip], hdulist[2].data[chip], bounds_error=False, fill_value=np.nan)
            chip_error.append(f(apogee_wave))
            f = interp1d(hdulist[4].data[chip], hdulist[5].data[chip], bounds_error=False, fill_value=np.nan)
            chip_sky.append(f(apogee_wave))

            bitmasks = utilities.identify_bitmasks(hdulist[3].data[chip])
            f = interp1d(hdulist[4].data[chip], bitmasks, bounds_error=False, kind='nearest', fill_value=0)
            chip_mask.append(f(apogee_wave))

        julian_date = hdulist[0].header['JD-MID']
        calc_bary_corr = helcorr(apo_longitude, apo_latitude, apo_altitude, apStar['ras'], apStar['dec'], julian_date)
        hdulist.close()

        apStar['flux'].append(np.nansum(np.array(chip_flux), axis=0))
        apStar['error'].append(np.nansum(np.array(chip_error), axis=0))
        apStar['sky'].append(np.nansum(np.array(chip_sky), axis=0))
        apStar['mask'].append(np.nansum(np.array(chip_mask), axis=0).astype(bool))
        apStar['bcs'] = np.append(apStar['bcs'], -1*calc_bary_corr[0])

    apStar['flux'] = np.array(apStar['flux'])
    apStar['error'] = np.array(apStar['error'])
    apStar['sky'] = np.array(apStar['sky'])
    apStar['mask'] = np.array(apStar['mask'], dtype=bool)

    """SOMETIMES THERE ARE VISITS THAT HAVE NO DATA? REMOVE THEM"""
    empty_columns = False

    if np.shape(apStar['flux'])[0] == 1 and np.all(apStar['flux'] == 0):
        empty_columns = True
        
    elif np.shape(apStar['flux'])[0] > 1 and np.all(np.prod(apStar['flux'], axis=0) == 0): # if there is at least one visit that is all zeros
        empty_columns = True

    if empty_columns:
        print('ARRR HERE THERE BE EMPTY COLUMNS')
        
        nonzero_cols = np.ones((apStar['nvis'],), dtype=bool)
        nonzero_cols[np.where(np.sum(apStar['flux'], axis=1) == 0)[0]] = False

        apStar['flux'] = apStar['flux'][nonzero_cols]
        apStar['error'] = apStar['error'][nonzero_cols]
        apStar['sky'] = apStar['sky'][nonzero_cols]
        apStar['mask'] = apStar['mask'][nonzero_cols].astype(bool)
        apStar['fibers'] = apStar['fibers'][nonzero_cols]
        apStar['rvs'] = apStar['rvs'][nonzero_cols]
        apStar['bcs'] = apStar['bcs'][nonzero_cols]
        apStar['chips'] = apStar['chips'][nonzero_cols]
        apStar['nvis'] = len(apStar['flux'])
    
    for visit in range(apStar['nvis']):
        ### FIND WHERE THE CHIPS END
        apStar['chips'][visit] = utilities.identify_chip_ends(apStar['flux'][visit], fix_chip_gaps)

    return apStar

###########################################################################
###########################################################################
###########################################################################
"""This function stacks spectra from multiple fibers together, built from apVisit"""
###########################################################################

def stack_fiber_spectra_from_apVisit(chip_apStar, stack_type, mask_lim=2):

    chip_apStar['rvs'] = np.squeeze(chip_apStar['rvs']) if chip_apStar['rvs'].ndim != 1 else chip_apStar['rvs']
    new_wave = []
    starts, ends = [], []
        
    for nn in range(chip_apStar['nvis']):
        new_wave.append([x - x*chip_apStar['rvs'][nn]/299792.458 for x in chip_apStar['wave'][nn]])
        starts.append(chip_apStar['wave'][nn][0])
        starts.append(new_wave[-1][0])
        ends.append(chip_apStar['wave'][nn][-1])
        ends.append(new_wave[-1][-1])

    wave_inds = np.where((apogee_wave >= np.max(starts)) & (apogee_wave <= np.min(ends)))[0]
    interp_wave = apogee_wave[wave_inds]
    
    """interpolate EVERYTHING(?) onto new wavelength array, except sky and mask, since those are in rest frame, right?"""
    
    interp_flux, interp_nois, interp_mask, interp_cont, interp_skyy = [], [], [], [], []
    
    for ii in range(chip_apStar['nvis']):
        
        f = interp1d(new_wave[ii], chip_apStar['flux'][ii], bounds_error=False, fill_value = np.nan)
        interp_flux.append(f(interp_wave))
        
        f = interp1d(new_wave[ii], chip_apStar['error'][ii], bounds_error=False, fill_value = np.nan)
        interp_nois.append(f(interp_wave))
        
        f = interp1d(new_wave[ii], chip_apStar['cont'][ii], bounds_error=False, fill_value = np.nan)
        interp_cont.append(f(interp_wave))
        
        f = interp1d(new_wave[ii], chip_apStar['mask'][ii], kind='nearest', bounds_error=False, fill_value = True)
        interp_mask.append(f(interp_wave).astype(bool))

        f = interp1d(chip_apStar['wave'][ii], chip_apStar['sky'][ii], bounds_error=False, fill_value = np.nan)
        interp_skyy.append(f(interp_wave))

    interp_flux, interp_nois, interp_cont = np.array(interp_flux), np.array(interp_nois), np.array(interp_cont)
    interp_mask, interp_skyy = np.array(interp_mask, dtype=bool), np.array(interp_skyy)

    """ INITIALIZE RETURN DICTIONARY """
    shape = np.shape(interp_wave)
    return_chip = dict(wave = interp_wave, flux = np.zeros(shape), error = np.zeros(shape), 
        mask = np.ones(shape, dtype=bool), sky = np.zeros(shape), cont = np.zeros(shape))

    return_chip['lim'] = np.array([wave_inds[0], wave_inds[-1]+1], dtype=int)

    if stack_type == 'median':

        for pixel in range(len(interp_wave)):
            
            """ SKY FLUX AND CONTINUUM ARE AVERAGED OVER ALL VISITS REGARDLESS OF MASKS """
    
            if len(np.where(interp_mask[:,pixel] == False)[0]) > chip_apStar['nvis']/mask_lim:
    
                return_chip['mask'][pixel] = False
            
            return_chip['sky'][pixel] = np.nanmean(interp_skyy[:,pixel])
            return_chip['cont'][pixel] = np.nanmean(interp_cont[:,pixel])
    
            return_chip['flux'][pixel] = np.nanmedian(interp_flux[interp_mask[:,pixel],pixel])
            return_chip['error'][pixel] = np.nanmedian(interp_nois[interp_mask[:,pixel],pixel])
    
            if np.isnan(return_chip['flux'][pixel]):
                return_chip['mask'][pixel] = False

    if stack_type == 'invvar':
        for pixel in range(len(return_chip['wave'])):
    
            """ SKY FLUX AND CONTINUUM ARE AVERAGED OVER ALL VISITS REGARDLESS OF MASKS """
            return_chip['sky'][pixel] = np.nanmean(chip_apStar['sky'][:,pixel])
            return_chip['cont'][pixel] = np.nanmean(chip_apStar['cont'][:,pixel])
    
            """ IF A PIXEL IS MASKED IN AT LEAST HALF OF THE VISITS, IT'S MASKED IN THE STACKED SPECTRUM,
                BUT THE FLUX AND ERROR ARRAYS ARE CALCULATED SANS MASKS """
            if len(np.where(chip_apStar['mask'][:,pixel] == False)[0]) > chip_apStar['nvis']/mask_lim:
    
                numerator = np.sum(chip_apStar['flux'][:,pixel]/
                                   chip_apStar['error'][:,pixel]**2)
                denominator = np.sum(1/chip_apStar['error'][:,pixel]**2)
    
                return_chip['flux'][pixel] = numerator / denominator
                return_chip['error'][pixel] = 1 / denominator
    
                return_chip['mask'][pixel] = False
    
            else:
                """ OTHERWISE, THE FLUX AND ERROR IN THAT PIXEL IS CALCULATED 
                EXCLUDING PIXELS MASKED IN THE VISITS """
                
                numerator = np.sum(chip_apStar['flux'][chip_apStar['mask'][:,pixel],pixel]/
                                   chip_apStar['error'][chip_apStar['mask'][:,pixel],pixel]**2)
                denominator = np.sum(1/chip_apStar['error'][chip_apStar['mask'][:,pixel],pixel]**2)
        
                return_chip['flux'][pixel] = numerator / denominator
                return_chip['error'][pixel] = 1 / denominator
    
            if np.isnan(return_chip['flux'][pixel]):
                return_chip['mask'][pixel] = False

    # return np.vstack([return_chip['wave'], return_chip['flux'], return_chip['error'], return_chip['mask'].astype(bool), return_chip['cont'], return_chip['sky']])
    return return_chip

# def stack_fiber_spectra_from_apVisit(chip_apStar, stack_type, mask_lim=2):
#     chip_apStar['rvs'] = np.squeeze(chip_apStar['rvs']) if chip_apStar['rvs'].ndim != 1 else chip_apStar['rvs']
#     new_wave, starts, ends = [], [], []
    
#     for nn in range(chip_apStar['nvis']):
#         new_wave.append([x - x*chip_apStar['rvs'][nn]/299792.458 for x in chip_apStar['wave'][nn]])
#         starts.append(chip_apStar['wave'][nn][0])
#         starts.append(new_wave[-1][0])
#         ends.append(chip_apStar['wave'][nn][-1])
#         ends.append(new_wave[-1][-1])

    

#     wave_inds = np.where((apogee_wave >= np.max(starts)) & (apogee_wave <= np.min(ends)))[0]
#     interp_wave = apogee_wave[wave_inds]
    

#     # Interpolation
#     interp_flux, interp_nois, interp_mask, interp_cont, interp_skyy = [], [], [], [], []
#     for ii in range(chip_apStar['nvis']):
#         for key, data in zip(['flux', 'nois', 'cont', 'mask', 'skyy'], 
#                              [chip_apStar['flux'], chip_apStar['error'], chip_apStar['cont'], chip_apStar['mask'], chip_apStar['sky']]):
#             f = interp1d(new_wave[ii], data[ii], bounds_error=False, fill_value=np.nan if key != 'mask' else True, kind='nearest' if key == 'mask' else 'linear')
#             locals()[f"interp_{key}"].append(f(interp_wave))
    
#     interp_flux, interp_nois, interp_cont = np.array(interp_flux), np.array(interp_nois), np.array(interp_cont)
#     interp_mask, interp_skyy = np.array(interp_mask, dtype=bool), np.array(interp_skyy)

#     return_chip = {key: np.zeros(np.shape(interp_wave)) if key not in ['mask'] else np.ones(np.shape(interp_wave), dtype=bool) 
#                    for key in ['flux', 'error', 'mask', 'sky', 'cont']}
#     return_chip['wave'] = np.copy(interp_wave)
#     return_chip['lim'] = np.array([wave_inds[0], wave_inds[-1]+1], dtype=int)
    
#     for pixel in range(len(interp_wave)):
#         if stack_type == 'median':
#             mask_count = len(np.where(interp_mask[:, pixel] == False)[0])
#             if mask_count > chip_apStar['nvis'] / mask_lim:
#                 return_chip['mask'][pixel] = False
#             return_chip['sky'][pixel] = np.nanmean(interp_skyy[:, pixel])
#             return_chip['cont'][pixel] = np.nanmean(interp_cont[:, pixel])
#             return_chip['flux'][pixel] = np.nanmedian(interp_flux[interp_mask[:, pixel], pixel])
#             return_chip['error'][pixel] = np.nanmedian(interp_nois[interp_mask[:, pixel], pixel])
#             if np.isnan(return_chip['flux'][pixel]):
#                 return_chip['mask'][pixel] = False

#         elif stack_type == 'invvar':
#             return_chip['sky'][pixel] = np.nanmean(chip_apStar['sky'][:, pixel])
#             return_chip['cont'][pixel] = np.nanmean(chip_apStar['cont'][:, pixel])
#             mask_count = len(np.where(chip_apStar['mask'][:, pixel] == False)[0])
#             if mask_count > chip_apStar['nvis'] / mask_lim:
#                 numerator = np.sum(chip_apStar['flux'][:, pixel] / chip_apStar['error'][:, pixel]**2)
#                 denominator = np.sum(1 / chip_apStar['error'][:, pixel]**2)
#                 return_chip['flux'][pixel] = numerator / denominator
#                 return_chip['error'][pixel] = 1 / denominator
#                 return_chip['mask'][pixel] = False
#             else:
#                 numerator = np.sum(chip_apStar['flux'][chip_apStar['mask'][:, pixel], pixel] / chip_apStar['error'][chip_apStar['mask'][:, pixel], pixel]**2)
#                 denominator = np.sum(1 / chip_apStar['error'][chip_apStar['mask'][:, pixel], pixel]**2)
#                 return_chip['flux'][pixel] = numerator / denominator
#                 return_chip['error'][pixel] = 1 / denominator
#             if np.isnan(return_chip['flux'][pixel]):
#                 return_chip['mask'][pixel] = False

#     return return_chip


###########################################################################
###########################################################################
###########################################################################
"""This function merges two apStar files in the cases where there are multiple corresponding to the same OBJID"""
###########################################################################

def merge_apStar(dict_1, dict_2):
    
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_1.items():
        if key != 'nvis':
            dict_3[key] = np.concatenate([dict_1[key], dict_2[key]])
        else:
            dict_3[key] = dict_1[key] + dict_2[key]
            
    return dict_3


###########################################################################
###########################################################################
###########################################################################
"""This function co-adds fiber spectra"""
###########################################################################

def stack_fiber_spectra(chip_apStar):

    """ INITIALIZE RETURN DICTIONARY """
    shape = np.shape(chip_apStar['wave'])
    return_chip = dict(wave = chip_apStar['wave'], flux = np.zeros(shape), error = np.zeros(shape), 
                       mask = np.ones(shape, dtype=bool), sky = np.zeros(shape), cont = np.zeros(shape))

    for pixel in range(len(chip_apStar['wave'])):

        """ SKY FLUX AND CONTINUUM ARE AVERAGED OVER ALL VISITS REGARDLESS OF MASKS """

        if len(np.where(chip_apStar['mask'][:,pixel] == False)[0]) > chip_apStar['nvis']/2:

            return_chip['mask'][pixel] = False
        
        return_chip['sky'][pixel] = np.nanmean(chip_apStar['sky'][:,pixel])
        return_chip['cont'][pixel] = np.nanmean(chip_apStar['cont'][:,pixel])

        return_chip['flux'][pixel] = np.nanmedian(chip_apStar['flux'][chip_apStar['mask'][:,pixel],pixel])
        return_chip['error'][pixel] = np.nanmedian(chip_apStar['error'][chip_apStar['mask'][:,pixel],pixel])

        if np.isnan(return_chip['flux'][pixel]):
            return_chip['mask'][pixel] = False

    return return_chip

def collect_data_and_stack(file_paths, rv_shifts, chip):
    
    """I think I am going to collect the data and do the RV correction and interpolation onto the correct wavelength array while I am here."""
        
    new_wave, old_wave, old_flux, old_nois, old_mask, old_cont, old_skyy = [], [], [], [], [], [], []
    starts, ends = [], []
    this_id = []

    avg_rv_shift = np.nanmean(rv_shifts)
        
    for nn, path in enumerate(file_paths):
        hdu = fits.open(path)
        old_wave.append(hdu[chip].data[0])
        old_flux.append(hdu[chip].data[1])
        old_nois.append(hdu[chip].data[2])
        old_mask.append(hdu[chip].data[3])
        old_cont.append(hdu[chip].data[4])
        old_skyy.append(hdu[chip].data[5])
        this_id.append(hdu[0].header['ID'])
        hdu.close()
        

        new_wave.append([x - x*((avg_rv_shift-rv_shifts[nn])/299792.458) for x in old_wave[-1]])
        starts.append(old_wave[-1][0])
        starts.append(new_wave[-1][0])
        ends.append(old_wave[-1][-1])
        ends.append(new_wave[-1][-1])
        
        
    interp_wave = apogee_wave[(apogee_wave >= np.max(starts)) & (apogee_wave <= np.min(ends))]
    
    """interpolate EVERYTHING(?) onto new wavelength array, except sky and mask, since those are in rest frame, right?"""
    
    interp_flux, interp_nois, interp_mask, interp_cont, interp_skyy = [], [], [], [], []
    
    for ii in range(len(file_paths)):
        
        f = interp1d(new_wave[ii], old_flux[ii])
        interp_flux.append(f(interp_wave))
        
        f = interp1d(new_wave[ii], old_nois[ii])
        interp_nois.append(f(interp_wave))
        
        f = interp1d(new_wave[ii], old_cont[ii])
        interp_cont.append(f(interp_wave))
        
        f = interp1d(new_wave[ii], old_mask[ii], kind='nearest')
        interp_mask.append(f(interp_wave).astype(bool))
        
        interp_skyy.append(old_skyy[ii][(np.array(old_wave[ii]) >= np.max(starts)) & (np.array(old_wave[ii]) <= np.min(ends))])
        
    interp_flux, interp_nois, interp_cont = np.array(interp_flux), np.array(interp_nois), np.array(interp_cont)
    interp_mask, interp_skyy = np.array(interp_mask, dtype=bool), np.array(interp_skyy)
    
    average_sky = np.nanmean(interp_skyy, axis=0)
    
    spectra_to_stack = dict(wave=interp_wave, flux=interp_flux, error=interp_nois, mask=interp_mask, sky=interp_skyy, cont=interp_cont, nvis = len(file_paths))
    interp_stack = stack_fiber_spectra(spectra_to_stack)
    
    data = np.vstack((interp_stack['wave'], interp_stack['flux'], interp_stack['error'], interp_stack['mask'], interp_stack['cont'], 
                      interp_stack['sky']))

    
    return data
