#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import thecannon as tc
from ppxf.ppxf_util import log_rebin, convolve_gauss_hermite
from ppxf.ppxf import ppxf
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from multiprocessing import Process, Queue
from numba import jit


###########################################################################
"""Import all the necessary files and initialize directories"""
###########################################################################

homedir = '/uufs/astro.utah.edu/common/home/u1267339/'
alist_dir = '/uufs/astro.utah.edu/common/home/u1267339/ALIST/'

apogee_wave = np.loadtxt(homedir + 'InputData/lambdatemp.txt')




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
"""This function isolates the chosen chip of any spectrum or model"""
###########################################################################
    
def split_chips(spectrum_flux, spectrum_wave, chip):

    stops = identify_chip_ends(spectrum_flux)
        
    if chip == 'Blue':
        spec = spectrum_flux[stops[0]:stops[1]]
        wave = spectrum_wave[stops[0]:stops[1]]

    if chip == 'Green':
        spec = spectrum_flux[stops[2]:stops[3]]
        wave = spectrum_wave[stops[2]:stops[3]]
        
    if chip == 'Red':
        spec = spectrum_flux[stops[4]:stops[5]]
        wave = spectrum_wave[stops[4]:stops[5]]
        
    return spec, wave


###########################################################################
###########################################################################
###########################################################################
"""This function identifies the index at which chips begin and end"""
###########################################################################

chip_gap_indices = [246, 3273, 3585, 6079, 6344, 8334]

def identify_chip_ends(spectrum, fix_chip_gaps=False):

    """IDENTIFY THE INDEX AT WHICH THE CHIPS STOP AND START"""

    spectrum[spectrum==1] = 0.0

    if fix_chip_gaps:
        return chip_gap_indices
    else:
        count, stops = 0, []
        for d in range(len(spectrum)):
            if spectrum[d] != 0.0 and count % 2 == 0:
                count += 1
                stops.append(int(d))
            if spectrum[d] == 0.0 and count % 2 == 1:
                count += 1
                stops.append(int(d-1))
        
        if len(stops) == 6:
            return stops
            
        else:
            real_stops = []
            stops = np.array(stops)
            for ss in chip_gap_indices:
                if np.argmin(np.abs(ss-stops)) < 100:
                    real_stops.append(stops[np.argmin(np.abs(ss-stops))])
                else:
                    real_stops.append(ss)
                
            return real_stops

    


###########################################################################
###########################################################################
###########################################################################
"""This function takes the running median of a spectrum ignoring masked pixels"""
###########################################################################

@jit(nopython=True) # JIT compilation for better performance
def running_median(flux, bin_width=500, mask=None):
    
    if mask is None:
        mask = np.ones(np.shape(flux), dtype=np.int8)
    # else:
    #     mask = np.asarray(mask, dtype=np.int8)
        
    running_median = np.zeros(np.shape(flux))

    # Manually pad the flux and mask arrays using slicing and concatenation
    pad_flux = np.empty(len(flux) + 2 * bin_width)
    pad_mask = np.empty(len(mask) + 2 * bin_width, dtype=np.bool_)
    
    # Fill the padded arrays with appropriate values
    pad_flux[:bin_width] = flux[bin_width-1::-1]  # Use the first bin_width values for the padding at the start
    pad_flux[bin_width:bin_width + len(flux)] = flux  # Fill the middle with the original flux
    pad_flux[bin_width + len(flux):] = flux[-1:-bin_width-1:-1]  # Use the last bin_width values for the padding at the end
    
    pad_mask[:bin_width] = mask[bin_width-1::-1]  # Use the first value for the mask padding at the start
    pad_mask[bin_width:bin_width + len(mask)] = mask  # Fill the middle with the original mask
    pad_mask[bin_width + len(mask):] = mask[-1:-bin_width-1:-1]  # Use the last value for the mask padding at the end
    
    for pixel in range(len(flux)):
        running_median[pixel] = np.nanmedian(pad_flux[pixel:pixel+2*bin_width+1][pad_mask[pixel:pixel+2*bin_width+1]])
        
    return running_median

# def running_median(flux, bin_width=500, mask=[0]):
    
#     if len(mask) == 1:
#         mask = np.ones(np.shape(flux), dtype=bool)
        
#     running_median = np.zeros(np.shape(flux))
    
#     pad_flux = np.pad(flux, bin_width, mode='reflect')
#     pad_mask = np.pad(mask, bin_width, mode='reflect')
    
#     for pixel in range(len(flux)):
#         inds = np.arange(pixel, pixel+2*bin_width)
#         running_median[pixel] = np.nanmedian(pad_flux[inds][pad_mask[inds]])
        
#     return running_median

###########################################################################
###########################################################################
###########################################################################
"""This function takes the running 95th percentile of a spectrum ignoring masked pixels
   Then does a 4th order polynomial fit to the 95th percentile to calculate the continuum"""
###########################################################################

def running_percentile(flux, wave, bin_width=500, mask=[0]):
    
    if len(mask) == 1:
        mask = np.ones(np.shape(flux), dtype=bool)
        
    running_median = np.zeros(np.shape(flux))
    
    pad_flux = np.pad(flux, bin_width, mode='reflect')
    pad_mask = np.pad(mask, bin_width, mode='reflect')
    
    for pixel in range(len(flux)):
        inds = np.arange(pixel, pixel+2*bin_width)
        running_median[pixel] = np.nanpercentile(pad_flux[inds][pad_mask[inds]], 95)

    polynomial_fit = np.polyfit(wave, running_median, deg=4)
    continuum = np.polyval(polynomial_fit, wave)
        
    return continuum

###########################################################################
###########################################################################
###########################################################################
"""This makes a continuum normalized version of a full apogee spectrum, ignoring chip gaps"""
###########################################################################

def continuum_norm_apogee_spectrum(spec, wave, method='median'):

    new_spec = np.copy(spec)
    
    chip_ends = identify_chip_ends(spec)

    if method == 'median':

        new_spec[chip_ends[0]:chip_ends[1]] *= 1/running_median(spec[chip_ends[0]:chip_ends[1]], bin_width=500)
        new_spec[chip_ends[2]:chip_ends[3]] *= 1/running_median(spec[chip_ends[2]:chip_ends[3]], bin_width=500)
        new_spec[chip_ends[4]:chip_ends[5]] *= 1/running_median(spec[chip_ends[4]:chip_ends[5]], bin_width=500)

    if method == 'percentile':

        new_spec[chip_ends[0]:chip_ends[1]] *= 1/running_percentile(spec[chip_ends[0]:chip_ends[1]], apogee_wave[chip_ends[0]:chip_ends[1]], bin_width=500)
        new_spec[chip_ends[2]:chip_ends[3]] *= 1/running_percentile(spec[chip_ends[2]:chip_ends[3]], apogee_wave[chip_ends[2]:chip_ends[3]], bin_width=500)
        new_spec[chip_ends[4]:chip_ends[5]] *= 1/running_percentile(spec[chip_ends[4]:chip_ends[5]], apogee_wave[chip_ends[4]:chip_ends[5]], bin_width=500)        
    
    return new_spec

###########################################################################
###########################################################################
###########################################################################
"""This function unshifts spectra to their rest frame, mainly used to remove barycentric corrections"""
###########################################################################

def unshift(flux, wave, vel, is_mask_array=False):

    kind = 'nearest' if is_mask_array else 'linear'
    value = False if is_mask_array else np.nan
        
    new_wave = [x - x*(vel/299792.458) for x in wave]
    f = interp1d(new_wave, flux, bounds_error=False, kind=kind, fill_value=np.nan)
    shift_flux = f(wave)
    
    return shift_flux



###########################################################################
###########################################################################
###########################################################################
"""This function flags pixels affected by various APOGEE pixel-level bitmasks"""
###########################################################################

bits_to_mask = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]

def identify_bitmasks(masks):
        
    # initialize arrays that will say if a pixel has a given mask
    bitmask = np.zeros((len(bits_to_mask), len(masks)))

    # run through all the bits
    for l in range(len(bits_to_mask)):
        bitmask[l] = (np.bitwise_and(masks, 2**bits_to_mask[l]) != 0) # this identifies if a pixel has mask l applied
    
    # this is the number of different masks applied to each pixel. 
    # since it is boolean though, we only care if it is zero or not zero.
    pixmask = np.sum(bitmask, axis=0, dtype=bool)
    
    return ~pixmask


###########################################################################
###########################################################################
###########################################################################
"""This function identifies skylines in spectra using the object flux and sky flux"""
###########################################################################

def identify_skylines_and_continuum(wave, flux, sky, fiber, bin_width, renormalize=True, spec_type='galaxy'):
    
    # vERIFY CORRECT BIN SIZE
    if bin_width % 2 == 0:
        bin_width = bin_width/2
    else:
        print('Invalid Bin Size, choose even number')
    bin_width = int(bin_width)

    continuum = running_median(flux, bin_width)

    sky_flux = np.abs(sky)
    delta_flux = np.abs(flux - continuum)

    if spec_type == 'galaxy':
        delta_flux_cutoff = np.nanpercentile(delta_flux, 85)
        sky_flux_cutoff = np.nanpercentile(sky_flux, 60)

    if spec_type == 'cluster':
        delta_flux_cutoff = np.nanpercentile(delta_flux, 90)
        sky_flux_cutoff = np.nanpercentile(sky_flux, 85)
        
    if spec_type == 'star':
        delta_flux_cutoff = np.nanpercentile(delta_flux, 99)
        sky_flux_cutoff = np.nanpercentile(sky_flux, 99)

    skyline_mask = np.ones(np.shape(flux), dtype=bool)

    skyline_mask = np.ones(np.shape(flux), dtype=bool)
    skyline_mask[[0, -1]] = False

    for I in range(len(flux)-1):

        if fiber >= 20 and fiber <= 35 and wave[I] >= 15890 and wave[I] <= 15960:
            skyline_mask[I] = False
        
        elif sky_flux[I] > sky_flux_cutoff and delta_flux[I] > delta_flux_cutoff:
            skyline_mask[I-1:I+1] = False

        elif flux[I]/continuum[I] < -1 or flux[I]/continuum[I] > 3  or np.isnan(flux[I]/continuum[I]):
            skyline_mask[I] = False

    if renormalize:
        continuum = running_median(flux, bin_width, skyline_mask.astype(np.int8))

    return skyline_mask, continuum


###########################################################################
###########################################################################
"""This function log-rebins a cannon model spectrum to prep it for fitting"""
###########################################################################

def log_rebin_cannon(chosen_params, data_velscale, data_first_log_wave, data_lin_wave, single_template=True):
    
    # must add a second axis so that chosen_params has the right shape for single templates.  (1,3) vs. (3,)
    if single_template:
        chosen_params = chosen_params[np.newaxis]
    
    for nn in range(len(chosen_params)):
        
        # generate the cannon model at chosen_params
        test_design_matrix = model.vectorizer((chosen_params[nn] - model._fiducials)/model._scales)
        spec = np.dot(model.theta, test_design_matrix)[:,0]  
        spec[spec==0] = 1. # set chip gaps equal to 1

        # interpolate cannon template onto a linear wavelength array
        f = interp1d(apogee_wave, spec)
        lin_spec = f(data_lin_wave)
        
        # then log rebin using the velscale from the data spectrum
        wave_range = [np.min(data_lin_wave), np.max(data_lin_wave)]
        template, log_rebinned_wave, velscale_ = log_rebin(wave_range, lin_spec, velscale = data_velscale)
        
        template = template/np.nanmedian(template)
        
        # this just saves all of them in a 2D array
        if nn == 0:
            all_templates = np.copy(template)
        else:
            all_templates = np.vstack([all_templates, template])

    # then calculate the differential velocity betweent the first wavelength of the data spectrum and the template.
    dv = (log_rebinned_wave[0] - data_first_log_wave)*299792.458

    # return the templates and the differential velocity
    return all_templates, dv



###########################################################################
###########################################################################
"""This function log-rebins a standard model spectrum to prep it for fitting"""
###########################################################################

def log_rebin_model(chosen_templates, data_velscale, data_first_log_wave, data_lin_wave, single_template=True):
    
    # must add a second axis so that chosen_params has the right shape for single templates.  (1,3) vs. (3,)
    if single_template:
        chosen_template = chosen_template[np.newaxis]
    
    for nn in range(len(chosen_templates)):

        spec = np.copy(chosen_templates[nn])

        # interpolate cannon template onto a linear wavelength array
        f = interp1d(apogee_wave, spec)
        lin_spec = f(data_lin_wave)
        
        # then log rebin using the velscale from the data spectrum
        wave_range = [np.min(data_lin_wave), np.max(data_lin_wave)]
        template, log_rebinned_wave, velscale_ = log_rebin(wave_range, lin_spec, velscale = data_velscale)
        
        template = template/np.nanmedian(template)
        
        # this just saves all of them in a 2D array
        if nn == 0:
            all_templates = np.copy(template)
        else:
            all_templates = np.vstack([all_templates, template])

    # then calculate the differential velocity betweent the first wavelength of the data spectrum and the template.
    dv = (log_rebinned_wave[0] - data_first_log_wave)*299792.458

    # return the templates and the differential velocity
    return all_templates, dv


###########################################################################
###########################################################################
###########################################################################
"""This function performs the initial, high mdegree fit for error estimation"""
###########################################################################
    
def measure_error_scale(spec, templates, start, bound, mdeg=40, plot_this=False, return_kins=False, use_cannon=True):

    ###########################################################################
    """since this part is still done with pPXF, log rebin the spectrum and templates"""
    ###########################################################################
    
    # interpolate onto linearly sampled wavelength array
    lin_wave = np.linspace(spec['wav'][0], spec['wav'][-1], len(spec['wav']))
    f = interp1d(spec['wav'], spec['gal'])
    lin_galaxy = f(lin_wave)
    
    fn = interp1d(spec['wav'], spec['err'])
    lin_noise = fn(lin_wave)
    
    # interpolate onto logarithmically sampled wavelength array using a built-in ppxf function
    # do it first to calculate the velscale, then again to ensure the rebinning actually uses that velscale.
    wave_range = [spec['wav'][0], spec['wav'][-1]]
    log_galaxy, log_rebinned_wave, trash_velscale = log_rebin(wave_range, lin_galaxy, velscale=spec['vsc'])
    log_noise, log_rebinned_wave2, trash_velscale = log_rebin(wave_range, lin_noise, velscale=spec['vsc'])
    log_wave = np.exp(log_rebinned_wave) # rebinned galaxy wavelength array with real values in Angstroms

    log_noise /= np.nanmedian(log_galaxy[spec['pix']])
    log_galaxy /= np.nanmedian(log_galaxy[spec['pix']])     

    if use_cannon:
        temps, dv = log_rebin_cannon(templates, spec['vsc'], log_rebinned_wave[0], lin_wave, single_template=False)
    else:
        temps, dv = log_rebin_model(templates, spec['vsc'], log_rebinned_wave[0], lin_wave, single_template=False)
#     print("len(spec['gal'])", len(spec['gal']))
#     print("len(spec['pix'])", len(spec['pix']))
#     print('len(log_galaxy)', len(log_galaxy))
#     print('len(log_wave)', len(log_wave))

#     print('My velscale', spec['vsc'])
#     print("ppxf's velscale", trash_velscale)

    suppress_text = True
    if plot_this:
        suppress_text = False
    
    pp = ppxf(temps.T, log_galaxy, log_noise, spec['vsc'], start, bounds = bound, quiet=suppress_text,
               plot=False, moments=4, mdegree=mdeg, lam=log_wave, degree=-1, vsyst=dv, 
               goodpixels=spec['pix'], clean=False)
    
    if plot_this:
        plt.figure(figsize=(20,7))
        plt.plot(log_wave, log_galaxy, c='k', zorder=0)
        plt.plot(log_wave, pp.bestfit, c='r', zorder=2)
        plt.plot(log_wave, pp.mpoly, c='deepskyblue', zorder=3)
        plt.vlines(spec['wav'][spec['bad']], 0, 2, color='gray', zorder=1)
        plt.hlines(.8, log_wave[0], log_wave[-1], linestyle='dashed', color='k', zorder=2)
        plt.scatter(log_wave, log_galaxy-pp.bestfit + .8, marker='.', c='lime', zorder=0)
        plt.ylim(.7, 1.15)
        plt.show()
    
    resid = (log_galaxy[spec['pix']] - pp.bestfit[spec['pix']])
    scale = .7413*(np.nanpercentile(resid/log_noise[spec['pix']], 75) - 
                   np.nanpercentile(resid/log_noise[spec['pix']], 25))

    if return_kins:
        return scale, pp.sol
    else:
        return scale


###########################################################################
###########################################################################
###########################################################################
"""These two functions govern the process for getting accurate initial radial velocity estimates
   as well as getting a good measurment of the eSNR"""
###########################################################################

def fit_for_eSNR(blu_spec, gre_spec, red_spec, start_rv, rv_bound, disp_bound, alist_spec, output, use_cannon, mdeg):

    start = [start_rv, 1, 0, 0]
    bound = [[start_rv-rv_bound, start_rv+rv_bound], disp_bound, [-0.3, 0.3], [-0.3, 0.3]]

    Bscale, Bkin = measure_error_scale(blu_spec, alist_spec, start, bound, return_kins=True, use_cannon=use_cannon, mdeg=mdeg)
    Gscale, Gkin = measure_error_scale(gre_spec, alist_spec, start, bound, return_kins=True, use_cannon=use_cannon, mdeg=mdeg)
    Rscale, Rkin = measure_error_scale(red_spec, alist_spec, start, bound, return_kins=True, use_cannon=use_cannon, mdeg=mdeg)

    eSNR = np.nanmedian(np.concatenate((blu_spec['gal'][blu_spec['pix']]/(blu_spec['err'][blu_spec['pix']]*Bscale),      
        gre_spec['gal'][gre_spec['pix']]/(gre_spec['err'][gre_spec['pix']]*Gscale), 
        red_spec['gal'][red_spec['pix']]/(red_spec['err'][red_spec['pix']]*Rscale))))

    return output.put((eSNR, np.average([Bkin[0], Gkin[0], Rkin[0]]), np.average([Bkin[1], Gkin[1], Rkin[1]]), Bscale, Gscale, Rscale))

def calculate_eSNR_guess_RV(blu_spec, gre_spec, red_spec, rv_range, disp_bound, num_cpus, alist_spec, use_cannon=True, plot_this=True, mdeg=40):
    
    test_rvs = np.linspace(rv_range[0], rv_range[-1], num_cpus)
    rv_bound = np.average(np.diff(test_rvs))/2
    
    # Define an output queue
    output = Queue()
    
    # Setup a list of processes that we want to run
    processes = [Process(target=fit_for_eSNR, args=(blu_spec, gre_spec, red_spec, start_rv, rv_bound, disp_bound, alist_spec, output, use_cannon, mdeg)) for start_rv in test_rvs]
    
    # Run processes
    for p in processes:
        p.start()
    
    # Exit the completed processes
    for p in processes:
        p.join()
    
    # Get process results from the output queue
    results = np.array([output.get() for p in processes])

    best_results = results[np.argmax(results[:,0])]

    if plot_this:
        start = [best_results[1], best_results[2], 0, 0]
        bound = [[best_results[1]-1, best_results[1]+1], [best_results[2]-1, best_results[2]+1], [-.03, .03], [-.03, .03]]
        stuff = measure_error_scale(blu_spec, alist_spec, start, bound, return_kins=False, use_cannon=False, plot_this=True, mdeg=mdeg)
        # stuff = measure_error_scale(gre_spec, alist_spec, start, bound, return_kins=False, use_cannon=False, plot_this=True)
        # stuff = measure_error_scale(red_spec, alist_spec, start, bound, return_kins=False, use_cannon=False, plot_this=True)

    return best_results




###########################################################################
###########################################################################
###########################################################################
"""This function performs fits, iteratively increasing mdegree, until the fit stops improving"""
"""Then returns the multiplicative polynomial to use as the continuum for the MCMC fit"""
###########################################################################
    
def measure_continuum(spec, cannon_grid, start, bound, threshold=.01, normalize='median', plot_this=False):

    ###########################################################################
    """since this part is still done with pPXF, log rebin the spectrum and templates"""
    ###########################################################################
    
    # interpolate onto linearly sampled wavelength array
    lin_wave = np.linspace(spec['wav'][0], spec['wav'][-1], len(spec['wav']))
    f = interp1d(spec['wav'], spec['gal'])
    lin_galaxy = f(lin_wave)
    
    fn = interp1d(spec['wav'], spec['err'])
    lin_noise = fn(lin_wave)
    
    # interpolate onto logarithmically sampled wavelength array using a built-in ppxf function
    # do it first to calculate the velscale, then again to ensure the rebinning actually uses that velscale.
    wave_range = [spec['wav'][0], spec['wav'][-1]]
    log_galaxy, log_rebinned_wave, trash_velscale = log_rebin(wave_range, lin_galaxy, velscale=spec['vsc'])
    log_noise, log_rebinned_wave2, trash_velscale = log_rebin(wave_range, lin_noise, velscale=spec['vsc'])
    log_wave = np.exp(log_rebinned_wave) # rebinned galaxy wavelength array with real values in Angstroms
    
    if normalize == 'median':
        log_noise /= np.nanmedian(log_galaxy[spec['pix']])
        log_galaxy /= np.nanmedian(log_galaxy[spec['pix']])
    if normalize == 'percentile':
        log_noise /= np.nanpercentile(log_galaxy[spec['pix']], 95)
        log_galaxy /= np.nanpercentile(log_galaxy[spec['pix']], 95)        

    temps, dv = log_rebin_cannon(cannon_grid, spec['vsc'], log_rebinned_wave[0], lin_wave, normalize, single_template=False)
    
#     print("len(spec['gal'])", len(spec['gal']))
#     print("len(spec['pix'])", len(spec['pix']))
#     print('len(log_galaxy)', len(log_galaxy))
#     print('len(log_wave)', len(log_wave))

#     print('My velscale', spec['vsc'])
#     print("ppxf's velscale", trash_velscale)

    suppress_text = True
    if plot_this:
        suppress_text = False

    delta_chi2 = 100
    this_chi2 = 100
    mdeg = 5
    while np.abs(delta_chi2) > threshold and mdeg < 20:
        pp = ppxf(temps.T, log_galaxy, log_noise, spec['vsc'], start, bounds = bound, quiet=True,
                   plot=False, moments=2, mdegree=mdeg, lam=log_wave, degree=-1, vsyst=dv, 
                   goodpixels=spec['pix'], clean=False, bias=.7)

        delta_chi2 = this_chi2 - pp.chi2
        this_chi2 = pp.chi2
        mdeg += 1

        print('Mdegree: ', mdeg, ' Delta Chi2: ', delta_chi2, ' This chi2: ', this_chi2)
        
        if (plot_this and np.abs(delta_chi2) < threshold) or mdeg == 20:
            
            plt.figure(figsize=(20,7))
            plt.plot(log_wave, log_galaxy, c='k', zorder=0)
            plt.plot(log_wave, pp.bestfit, c='r', zorder=2)
            plt.plot(log_wave, pp.mpoly/np.nanmedian(pp.mpoly[spec['pix']]), c='deepskyblue', zorder=3)
            plt.vlines(spec['wav'][spec['bad']], 0, 2, color='gray', zorder=1)
            plt.hlines(.8, log_wave[0], log_wave[-1], linestyle='dashed', color='k', zorder=2)
            plt.scatter(log_wave, log_galaxy-pp.bestfit + .8, marker='.', c='lime', zorder=0)
            plt.ylim(.65, 1.15)
            plt.show()

    continuum = pp.mpoly[:-1] + np.diff(pp.mpoly)/2
    
    return continuum


###########################################################################
###########################################################################
###########################################################################
"""This function generates a shifted and broadened cannon template"""
###########################################################################

def shift_broaden_cannon(params, chip_ends, bin_width):
    ###########################################################################
    """generate cannon models and split chips"""
    """Shift, broaden, and combine templates, this is basically the same code as two_component_likelihood, btw."""
    ###########################################################################
    inputs = [params[2], params[3], np.log10(params[4]*1e9)]

    test_design_matrix = model.vectorizer((inputs - model._fiducials)/model._scales)
    spec = np.dot(model.theta, test_design_matrix)[:,0]
    
    # SPLIT CHIPS
    spec = spec[chip_ends[0]:chip_ends[1]]

    velscale = (299792.458*(np.log(apogee_wave[chip_ends[1]]) - np.log(apogee_wave[chip_ends[0]])))/len(spec)

    # SHIFT AND BROADEN TEMPLATE
    galaxy = convolve_gauss_hermite(spec-1, velscale, [params[0], params[1]], len(spec))+1

    galaxy /= median_filter(galaxy, bin_width)

    return galaxy



