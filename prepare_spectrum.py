#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import thecannon as tc
from ppxf.ppxf_util import convolve_gauss_hermite
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from DISSK import utilities
from astropy.io import fits


###########################################################################
"""Import all the necessary files and initialize directories"""
###########################################################################

homedir = '/uufs/astro.utah.edu/common/home/u1267339/'
alist_dir = '/uufs/astro.utah.edu/common/home/u1267339/ALIST/'

apogee_wave = np.loadtxt(homedir + 'InputData/lambdatemp.txt')
known_skylines = np.loadtxt(homedir + 'InputData/rousselot2000.dat')
known_skylines = known_skylines[known_skylines[:,1] > .5]
    

    
###########################################################################
###########################################################################
###########################################################################
"""This function reads in the cannon model, as it cannot be pickled for multiprocessing"""
###########################################################################

def initialize_cannon_model(model_name):
    """
    model_name is a string that points to the path location of the cannon model that will be used to analyze the data
    """
    global model
    try:
        model
    except NameError:
        model = tc.CannonModel.read(model_name)
    else:
        del model
        model = tc.CannonModel.read(model_name)
    
###########################################################################
###########################################################################
###########################################################################
"""This function prepares the spectra from my data reduction for fitting"""
###########################################################################

def prepare_spectrum(data, err_scale=None, median_normalize=True, mask_misfits=False, mask_known_skylines=False):
    
    ###########################################################################
    """read in wavelengths, spectrum, noise, mask, and continuum from files"""
    ###########################################################################

    galaxy = data[1]
    galaxy[np.isnan(galaxy)] = 0.
    noise = data[2]
    noise[np.isnan(noise)] = np.nanmax(noise)
    mask = data[3].astype(bool)
    wave = data[0]
    masked_waves = wave[~mask]
    continuum = data[4]
    
    if err_scale != None:
        noise *= err_scale
    
    velscale = (299792.458*(np.log(wave[-1]) - np.log(wave[0])))/len(wave)
    
    ###########################################################################
    """mask out all the pixels with fluxes we don't trust"""
    ###########################################################################
    
    wave_inds = np.where((apogee_wave >= wave[0]) & (apogee_wave <= wave[-1]))[0]
    chip_ends = np.array([wave_inds[0], wave_inds[-1]+1], dtype=int)

    return_mask = np.zeros(np.shape(wave))
    
    skylines = known_skylines[np.where((known_skylines[:,0] >= wave[0]) & (known_skylines[:,0] <= wave[-1]))[0]][:,0]
    ds = 2.5 # mask pixels within this many angstrom of a known skyline
    dl = .5 # mask pixels within this many angstrom of a masked pixel in mask

    # run through the whole wavelength array and mask out pixels that are:
    for i in range(len(wave)):
        
        # masked in the data file
        for j in range(len(masked_waves)):
            if wave[i] < masked_waves[j]+dl and wave[i] > masked_waves[j]-dl:
                return_mask[i] = 1
        
        """affected by skylines"""
        if mask_known_skylines:
            for j in range(len(skylines)):
                if wave[i] < skylines[j]+ds and wave[i] > skylines[j]-ds:
                    return_mask[i] = 1
                
        """consistently poorly fit by our models"""
        if mask_misfits:
            if (wave[i] < 15395 and wave[i] > 15360) or (wave[i] < 16070 and wave[i] > 16030) or (wave[i] < 16240 and wave[i] > 16210):
                return_mask[i] = 1
                
        # near to the chip gaps
        """this is how I've been doing it forever, but I want to try something different for a minute"""
#         if wave[i] > wave[-250] or wave[i] < wave[99]:
#             return_mask[i] = 1

        """masking stuff near the fixed template chip gaps"""
        if (wave[i] < apogee_wave[346]) or (wave[i] > apogee_wave[3024] and wave[i] < apogee_wave[3685]) or (wave[i] > apogee_wave[5830] and wave[i] < apogee_wave[6444]) or (wave[i] > apogee_wave[8085]):
        # if (wave[i] < apogee_wave[346]) or (wave[i] > apogee_wave[3024] and wave[i] < apogee_wave[6444]) or (wave[i] > apogee_wave[8085]):
            return_mask[i] = 1
            
#         # has nans in continuum???????????????????????
#         if np.isnan(float(galaxy[i])) == True or galaxy[i] <= 0:
#             return_mask[i] = 1
            
#         if np.isnan(float(noise[i])) == True or noise[i] <= 0 or noise[i] > 1e14:
#             return_mask[i] = 1
        
        # if galaxy[i] < .5 or galaxy[i] > 2:
        #     return_mask[i] = 1

    # good_pixels is a list of indices that have not been masked, so we have to convert return_mask from bool to indexes
    good_pixels = np.where(return_mask == 0)[0]
    bad_pixels = np.where(return_mask != 0)[0]
    
    # for ppxf the median value of the galaxy array has to be one
    if median_normalize:
        noise = noise/np.nanmedian(galaxy[good_pixels])
        galaxy = galaxy/np.nanmedian(galaxy[good_pixels])
    
#     class prepped_spec:
#         gal = galaxy
#         err = noise
#         wav = wave
#         pix = good_pixels
#         vsc = velscale1
#         lam = log_rebinned_wave[0]
#         lin = lin_wave

    prepped_spec = dict(gal = galaxy, err = noise, wav = wave, pix = good_pixels, vsc = velscale, cnt = continuum, bad = bad_pixels, lim = chip_ends)
            
    return prepped_spec


###########################################################################
###########################################################################
###########################################################################
"""This function prepares the spectra from my data reduction for fitting
   This version interpolates the spectra, error, continuum, noise, etc. onto the apogee wavelength array"""
###########################################################################

def prepare_spectrum_and_interp(data, err_scale):
    
    ###########################################################################
    """read in wavelengths, spectrum, noise, mask, and continuum from files"""
    ###########################################################################

    old_galaxy = data[1]
    old_galaxy[np.isnan(old_galaxy)] = 0.
    old_noise = data[2]
    old_noise[np.isnan(old_noise)] = np.nanmax(old_noise)
    old_mask = np.array(data[3], bool)
    old_wave = data[0]
    old_continuum = data[4]

    old_noise *= err_scale/old_continuum
    old_noise[old_continuum <= 0] = 1e15
    
    ###########################################################################
    """interpolate onto apogee wavelength array"""
    ########################################################################### 
    wave_inds = np.where((apogee_wave >= old_wave[0]) & (apogee_wave <= old_wave[-1]))[0]
    wave = apogee_wave[wave_inds]
    chip_ends = np.array([wave_inds[0], wave_inds[-1]+1], dtype=int)
    
    f = interp1d(old_wave, old_galaxy)
    galaxy = f(wave)
    
    f = interp1d(old_wave, old_noise)
    noise = f(wave)
    
    f = interp1d(old_wave, old_mask, kind='nearest')
    mask = f(wave).astype(bool)
    masked_waves = wave[mask]
    
    f = interp1d(old_wave, old_continuum)
    continuum = f(wave)
    
    velscale = np.array([(299792.458*(np.log(wave[-1]) - np.log(wave[0])))/len(wave)])
    
    ###########################################################################
    """mask out all the pixels with fluxes we don't trust"""
    ###########################################################################

    return_mask = np.zeros(np.shape(wave))
    
    skylines = known_skylines[np.where((known_skylines[:,0] >= wave[0]) & (known_skylines[:,0] <= wave[-1]))[0]][:,0]
    ds = 2.5 # mask pixels within this many angstrom of a known skyline
    dl = .5 # mask pixels within this many angstrom of a masked pixel in mask

    # run through the whole wavelength array and mask out pixels that are:
    for i in range(len(wave)):
        
        # affected by skylines
        for j in range(len(skylines)):
            if wave[i] < skylines[j]+ds and wave[i] > skylines[j]-ds:
                return_mask[i] = 1
        
        # masked in the data file
        for j in range(len(masked_waves)):
            if wave[i] < masked_waves[j]+dl and wave[i] > masked_waves[j]-dl:
                return_mask[i] = 1

        """masking stuff near the fixed template chip gaps"""
        if (wave[i] < apogee_wave[346]) or (wave[i] > apogee_wave[3024] and wave[i] < apogee_wave[3685]) or (wave[i] > apogee_wave[5830] and wave[i] < apogee_wave[6444]) or (wave[i] > apogee_wave[8085]):
            return_mask[i] = 1

    # good_pixels is a list of indices that have not been masked, so we have to convert return_mask from bool to indexes
    good_pixels = np.where(return_mask == 0)[0]
    bad_pixels = np.where(return_mask != 0)[0]
    
    # for ppxf the median value of the galaxy array has to be one
    noise = noise/np.nanmedian(galaxy[good_pixels])
    galaxy = galaxy/np.nanmedian(galaxy[good_pixels])

    prepped_spec = dict(gal = galaxy, err = noise, wav = wave, pix = good_pixels, vsc = velscale, cnt = continuum, bad = bad_pixels, lim = chip_ends)
            
    return prepped_spec


###########################################################################
###########################################################################  
###########################################################################
"""This function creates a two-component artificially noisy template"""
###########################################################################
    
def generate_two_component_ANT(params1, params2, chip, input_frac, input_snr, age, path, fiber_info, bin_width):
    
    """
    params1 and 2 are in the order [vel, disp, met, alpha, age(Gyr)]
    """
    
    if chip == 'Blue':
        cc = 1
        err_scale = fiber_info['B_sigmaG']
        
    if chip == 'Green':
        cc = 2
        err_scale = fiber_info['G_sigmaG']
        
    if chip == 'Red':
        cc = 3
        err_scale = fiber_info['R_sigmaG']
    
    
    
    ###########################################################################
    """read in wavelengths, spectrum, noise, mask, and continuum from files"""
    ###########################################################################
    
    
    
    hdu = fits.open(path)
    wave = hdu[cc].data[0]
#     this_flx = hdu[cc].data[1]
    chip_err = hdu[cc].data[2]
    chip_err[np.isnan(chip_err)] = np.nanmax(chip_err)
    chip_msk = hdu[cc].data[3].astype(bool)
    chip_sky = hdu[cc].data[5]
    masked_waves = wave[~chip_msk]
    
    hdu.close()
    
    chip_err *= err_scale/((input_snr-10)/fiber_info['eSNR'])
    
    velscale = (299792.458*(np.log(wave[-1]) - np.log(wave[0])))/len(wave)
    
    wave_inds = np.where((apogee_wave >= wave[0]) & (apogee_wave <= wave[-1]))[0]
    chip_ends = np.array([wave_inds[0], wave_inds[-1]+1], dtype=int)
    
    
    ###########################################################################
    """generate cannon models and split chips"""
    """Shift, broaden, and combine templates, this is basically the same code as two_component_likelihood, btw."""
    ###########################################################################
    inputs1 = [params1[2], params1[3], np.log10(age*1e9)]
    inputs2 = [params2[2], params2[3], np.log10(age*1e9)]

    test_design_matrix = model.vectorizer((inputs1 - model._fiducials)/model._scales)
    spec1 = np.dot(model.theta, test_design_matrix)[:,0]
    
    test_design_matrix = model.vectorizer((inputs2 - model._fiducials)/model._scales)
    spec2 = np.dot(model.theta, test_design_matrix)[:,0]
    
    # SPLIT CHIPS
    spec1 = spec1[wave_inds]
    spec2 = spec2[wave_inds]

    # SHIFT AND BROADEN TEMPLATE
    broad_spec1 = convolve_gauss_hermite(spec1-1, velscale, [params1[0], params1[1]], len(wave))+1
    broad_spec2 = convolve_gauss_hermite(spec2-1, velscale, [params2[0], params2[1]], len(wave))+1
    
    galaxy = (broad_spec1*input_frac) + (broad_spec2*(1-input_frac))
    
    
    ###########################################################################
    """add sky and noise, subtract sky"""
    ###########################################################################
    
    galaxy = galaxy + chip_sky
    
    for ii in range(len(galaxy)):
        galaxy[ii] += np.random.normal(loc=0, scale = chip_err[ii]*1.349)
        
    ### unsubtract skylines
    galaxy -= chip_sky
    
    ###############################################################################################################
    """identify and mask skylines and all the other bullshit"""
    ############################################################################################################### 
    
    skymask, new_pix_med, delta_flux_cutoff, sky_flux_cutoff = utilities.identify_skylines(wave, galaxy, chip_sky, 0, bin_width, renormalize=False)
    galaxy /= new_pix_med
    chip_err /= new_pix_med
    skyline_waves = wave[~skymask]

    
    # skylines = known_skylines[np.where((known_skylines[:,0] >= wave[0]) & (known_skylines[:,0] <= wave[-1]))[0]][:,0]
    # ds = 2.5 # mask pixels within this many angstrom of a known skyline

    ###########################################################################
    """mask out all the pixels with fluxes we don't trust"""
    ###########################################################################

    return_mask = np.zeros(np.shape(wave))
    dl = .5 # mask pixels within this many angstrom of a masked pixel in mask

    # run through the whole wavelength array and mask out pixels that are:
    for i in range(len(wave)):
        
        # affected by identified skylines
        for j in range(len(skyline_waves)):
            if wave[i] < skyline_waves[j]+dl and wave[i] > skyline_waves[j]-dl:
                return_mask[i] = 1
        
        # masked in the data file
        for j in range(len(masked_waves)):
            if wave[i] < masked_waves[j]+dl and wave[i] > masked_waves[j]-dl:
                return_mask[i] = 1

        """masking stuff near the fixed template chip gaps"""
        if (wave[i] < apogee_wave[346]) or (wave[i] > apogee_wave[3024] and wave[i] < apogee_wave[3685]) or (wave[i] > apogee_wave[5830] and wave[i] < apogee_wave[6444]) or (wave[i] > apogee_wave[8085]):
            return_mask[i] = 1

    galaxy[np.isnan(galaxy)] = 1

    # good_pixels is a list of indices that have not been masked, so we have to convert return_mask from bool to indexes
    good_pixels = np.where(return_mask == 0)[0]
    bad_pixels = np.where(return_mask != 0)[0]
    
    # for ppxf the median value of the galaxy array has to be one
    noise = chip_err/np.nanmedian(galaxy[good_pixels])
    galaxy = galaxy/np.nanmedian(galaxy[good_pixels])

    prepped_spec = dict(gal = galaxy, err = chip_err, wav = wave, pix = good_pixels, vsc = velscale, cnt = new_pix_med, bad = bad_pixels, lim = chip_ends)
            
    return prepped_spec

###########################################################################
###########################################################################  
###########################################################################
"""This function creates a two-component artificially noisy template"""
###########################################################################
    
def generate_one_component_ANT(params, chip, input_snr, path, fiber_info, bin_width):
    
    """
    params are in the order [vel, disp, met, alpha, age(Gyr)]
    """
    
    if chip == 'Blue':
        cc = 1
        err_scale = fiber_info['B_sigmaG']
        
    if chip == 'Green':
        cc = 2
        err_scale = fiber_info['G_sigmaG']
        
    if chip == 'Red':
        cc = 3
        err_scale = fiber_info['R_sigmaG']
    
    
    
    ###########################################################################
    """read in wavelengths, spectrum, noise, mask, and continuum from files"""
    ###########################################################################
    
    
    
    hdu = fits.open(path)
    wave = hdu[cc].data[0]
#     this_flx = hdu[cc].data[1]
    chip_err = hdu[cc].data[2]
    chip_err[np.isnan(chip_err)] = np.nanmax(chip_err)
    chip_msk = hdu[cc].data[3].astype(bool)
    chip_sky = hdu[cc].data[5]
#     masked_waves = wave[chip_msk]
    
    hdu.close()
    
    
    err_scale /= input_snr/fiber_info['eSNR']
    # err_scale = fiber_info['eSNR']/input_snr
    chip_err *= err_scale
    chip_err[chip_err <= 0] = 1e15
    
    velscale = (299792.458*(np.log(wave[-1]) - np.log(wave[0])))/len(wave)
    
    wave_inds = np.where((apogee_wave >= wave[0]) & (apogee_wave <= wave[-1]))[0]
    chip_ends = np.array([wave_inds[0], wave_inds[-1]+1], dtype=int)
    
    
    ###########################################################################
    """generate cannon models and split chips"""
    """Shift, broaden, and combine templates, this is basically the same code as two_component_likelihood, btw."""
    ###########################################################################
    inputs = [params[2], params[3], np.log10(params[4]*1e9)]

    test_design_matrix = model.vectorizer((inputs - model._fiducials)/model._scales)
    spec = np.dot(model.theta, test_design_matrix)[:,0]
    
    # SPLIT CHIPS
    spec = spec[wave_inds]

    # SHIFT AND BROADEN TEMPLATE
    galaxy = convolve_gauss_hermite(spec-1, velscale, [params[0], params[1]], len(wave))+1


    """for right now we're going to do no weird continuum adjustment, just running median normalization"""
    
    
    ###########################################################################
    """add sky and noise, subtract sky"""
    ###########################################################################
    
    galaxy = galaxy + chip_sky
    
    for ii in range(len(galaxy)):
        galaxy[ii] += np.random.normal(loc=0, scale = chip_err[ii]*1.349)
        
    ### unsubtract skylines
    galaxy -= chip_sky
    
    ###############################################################################################################
    """identify and mask skylines and all the other bullshit"""
    ############################################################################################################### 
    
    skymask, new_pix_med = utilities.identify_skylines(wave, galaxy, chip_sky, 0, bin_width)
    galaxy /= new_pix_med
    full_mask = chip_msk + skymask
    masked_waves = wave[full_mask]
#     skylines = known_skylines[np.where((known_skylines[:,0] >= wave[0]) & (known_skylines[:,0] <= wave[-1]))[0]][:,0]
#     ds = 2.5 # mask pixels within this many angstrom of a known skyline

    ###########################################################################
    """mask out all the pixels with fluxes we don't trust"""
    ###########################################################################

    return_mask = np.zeros(np.shape(wave))
    dl = .5 # mask pixels within this many angstrom of a masked pixel in mask

    # run through the whole wavelength array and mask out pixels that are:
    for i in range(len(wave)):
        
        # affected by skylines
#         for j in range(len(skylines)):
#             if wave[i] < skylines[j]+ds and wave[i] > skylines[j]-ds:
#                 return_mask[i] = 1
        
        # masked in the data file
        for j in range(len(masked_waves)):
            if wave[i] < masked_waves[j]+dl and wave[i] > masked_waves[j]-dl:
                return_mask[i] = 1

        """masking stuff near the fixed template chip gaps"""
        if (wave[i] < apogee_wave[346]) or (wave[i] > apogee_wave[3024] and wave[i] < apogee_wave[3685]) or (wave[i] > apogee_wave[5830] and wave[i] < apogee_wave[6444]) or (wave[i] > apogee_wave[8085]):
            return_mask[i] = 1

    # good_pixels is a list of indices that have not been masked, so we have to convert return_mask from bool to indexes
    good_pixels = np.where(return_mask == 0)[0]
    bad_pixels = np.where(return_mask != 0)[0]
    
    # for ppxf the median value of the galaxy array has to be one
    noise = chip_err/np.nanmedian(galaxy[good_pixels])
    galaxy = galaxy/np.nanmedian(galaxy[good_pixels])

    prepped_spec = dict(gal = galaxy, err = chip_err, wav = wave, pix = good_pixels, vsc = velscale, cnt = new_pix_med, bad = bad_pixels, lim = chip_ends)
            
    return prepped_spec



    
    
