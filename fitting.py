#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import thecannon as tc
from ppxf.ppxf_util import convolve_gauss_hermite
from multiprocessing import Pool
import emcee
from tqdm import tqdm
import corner
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from . import plot_and_save, utilities


###########################################################################
"""Import all the necessary files and initialize directories"""
###########################################################################

homedir = '/uufs/astro.utah.edu/common/home/u1267339/'
alist_dir = '/uufs/astro.utah.edu/common/home/u1267339/ALIST/'

apogee_wave = np.loadtxt(homedir + 'InputData/lambdatemp.txt')

mean_velocity_offset = 12


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
    global hull
    try:
        model
    except NameError:
        model = tc.CannonModel.read(model_name)
        hull = Delaunay(model.training_set_labels)
    else:
        del model
        del hull
        model = tc.CannonModel.read(model_name)
        hull = Delaunay(model.training_set_labels)

def initialize_second_cannon_model(model_name):
    """
    model_name is a string that points to the path location of the cannon model that will be used to analyze the data
    """
    global model2
    global hull2
    try:
        model2
    except NameError:
        model2 = tc.CannonModel.read(model_name)
        hull2 = Delaunay(model2.training_set_labels)
    else:
        del model2
        del hull2
        model2 = tc.CannonModel.read(model_name)
        hull2 = Delaunay(model2.training_set_labels)


###########################################################################
###########################################################################  
###########################################################################
"""This function uses the above function to generate a model to fit to the data for each chip"""
###########################################################################

def generate_model(template_params, kinematics, bin_width, Bprepped, Gprepped, Rprepped):
    # generate the cannon template
    test_design_matrix = model.vectorizer((template_params - model._fiducials)/model._scales)
    template = np.dot(model.theta, test_design_matrix)[:,0]  

    # SPLIT CHIPS
    blu_temp = template[Bprepped['lim'][0]:Bprepped['lim'][1]]
    gre_temp = template[Gprepped['lim'][0]:Gprepped['lim'][1]]
    red_temp = template[Rprepped['lim'][0]:Rprepped['lim'][1]]

    blu_broad_temp = convolve_gauss_hermite(blu_temp-1, Bprepped['vsc'], kinematics, len(blu_temp))+1
    gre_broad_temp = convolve_gauss_hermite(gre_temp-1, Gprepped['vsc'], kinematics, len(gre_temp))+1
    red_broad_temp = convolve_gauss_hermite(red_temp-1, Rprepped['vsc'], kinematics, len(red_temp))+1

    blu_model = blu_broad_temp/median_filter(blu_broad_temp, bin_width)
    gre_model = gre_broad_temp/median_filter(gre_broad_temp, bin_width)
    red_model = red_broad_temp/median_filter(red_broad_temp, bin_width)

    # blu_model = blu_broad_temp/utilities.running_median(blu_broad_temp, int(bin_width/2))
    # gre_model = gre_broad_temp/utilities.running_median(gre_broad_temp, int(bin_width/2))
    # red_model = red_broad_temp/utilities.running_median(red_broad_temp, int(bin_width/2))


    return blu_model, gre_model, red_model

def generate_model2(template_params, kinematics, bin_width, Bprepped, Gprepped, Rprepped):
    # generate the cannon template
    test_design_matrix = model2.vectorizer((template_params - model2._fiducials)/model2._scales)
    template = np.dot(model2.theta, test_design_matrix)[:,0]  

    # SPLIT CHIPS
    blu_temp = template[Bprepped['lim'][0]:Bprepped['lim'][1]]
    gre_temp = template[Gprepped['lim'][0]:Gprepped['lim'][1]]
    red_temp = template[Rprepped['lim'][0]:Rprepped['lim'][1]]

    blu_broad_temp = convolve_gauss_hermite(blu_temp-1, Bprepped['vsc'], kinematics, len(blu_temp))+1
    gre_broad_temp = convolve_gauss_hermite(gre_temp-1, Gprepped['vsc'], kinematics, len(gre_temp))+1
    red_broad_temp = convolve_gauss_hermite(red_temp-1, Rprepped['vsc'], kinematics, len(red_temp))+1

    blu_model = blu_broad_temp/median_filter(blu_broad_temp, bin_width)
    gre_model = gre_broad_temp/median_filter(gre_broad_temp, bin_width)
    red_model = red_broad_temp/median_filter(red_broad_temp, bin_width)

    return blu_model, gre_model, red_model



###########################################################################
###########################################################################  
###########################################################################
"""This function evaluates the chi2 of a model spectrum to the data"""
###########################################################################

def evaluate_chi2(model, spectrum, good_pix, error):

    return np.sum((spectrum[good_pix] - model[good_pix])**2 / error[good_pix]**2)


###########################################################################
###########################################################################  
###########################################################################
"""This function evaluates the log likelihood value for each one-component model"""
###########################################################################

def evaluate_onecomp_likelihood(kinematics, Bprepped, Gprepped, Rprepped, template_params, bin_width):

    blu_model, gre_model, red_model = generate_model(template_params, kinematics, bin_width, Bprepped, Gprepped, Rprepped)

    
    # CALCULATE CHI SQUARED
    Bchi2 = evaluate_chi2(blu_model, Bprepped['gal'], Bprepped['pix'], Bprepped['err'])
    Gchi2 = evaluate_chi2(gre_model, Gprepped['gal'], Gprepped['pix'], Gprepped['err'])
    Rchi2 = evaluate_chi2(red_model, Rprepped['gal'], Rprepped['pix'], Rprepped['err'])
    
    likelihood = -1*(Bchi2 + Gchi2 + Rchi2)/2

    return likelihood


###########################################################################
###########################################################################  
###########################################################################
"""This function evaluates the log likelihood value for each one-component model"""
###########################################################################

def initialize_walkers_with_convex_hull(fitting_parameters, fit_type='one_component'):

    initial = np.zeros((fitting_parameters['walkers'], fitting_parameters['dimensions']))
    if fit_type == 'one_component':
        for ii in [0,1]:
            initial[:,ii] = np.random.uniform(fitting_parameters['start_bounds'][ii][0], fitting_parameters['start_bounds'][ii][1], fitting_parameters['walkers']) 
            
        count = 0
        while count < fitting_parameters['walkers']:
            m = np.random.uniform(fitting_parameters['start_bounds'][2][0], fitting_parameters['start_bounds'][2][1])
            a = np.random.uniform(fitting_parameters['start_bounds'][3][0], fitting_parameters['start_bounds'][3][1])
            if fitting_parameters['set_age']:
                if hull.find_simplex([m, a, np.log10(fitting_parameters['set_age']*1e9)]) >= 0:
                    initial[count,2] = m
                    initial[count,3] = a
                    count += 1
            else:
                A = np.random.uniform(fitting_parameters['start_bounds'][4][0], fitting_parameters['start_bounds'][4][1])
                if hull.find_simplex([m, a, np.log10(A*1e9)]) >= 0:
                    initial[count,2] = m
                    initial[count,3] = a
                    initial[count,4] = A
                    count += 1

    if fit_type == 'two_components':
        count = 0
        # print('initializing velocities')
        while count < fitting_parameters['walkers']:
            v1 = np.random.uniform(fitting_parameters['start_bounds'][0][0], fitting_parameters['start_bounds'][0][1])
            v2 = np.random.uniform(fitting_parameters['start_bounds'][1][0], fitting_parameters['start_bounds'][1][1])
            vf = (fitting_parameters['set_frac']**2)*v1 + (1 - fitting_parameters['set_frac']**2)*v2
            if np.sign(fitting_parameters['vel'] - v1) != np.sign(fitting_parameters['vel'] - v2) and np.abs(v1 + 300) < np.abs(v2 + 300) and vf >= fitting_parameters['vel'] - mean_velocity_offset and vf <= fitting_parameters['vel'] + mean_velocity_offset:
                initial[count,0] = v1
                initial[count,1] = v2
                count += 1

        count = 0
        while count < fitting_parameters['walkers']:
            d1 = np.random.uniform(fitting_parameters['start_bounds'][2][0], fitting_parameters['start_bounds'][2][1])
            d2 = np.random.uniform(fitting_parameters['start_bounds'][3][0], fitting_parameters['start_bounds'][3][1])
            if fitting_parameters['disp_prior'] == 'standard' and d1 > d2 or fitting_parameters['disp_prior'] == 'reversed' and d1 < d2:
                initial[count,2] = d1
                initial[count,3] = d2
                count += 1   

        if not fitting_parameters['set_params']:
            count = 0
            # print('initializing params1')
            while count < fitting_parameters['walkers']:
                m = np.random.uniform(fitting_parameters['start_bounds'][4][0], fitting_parameters['start_bounds'][4][1])
                a = np.random.uniform(fitting_parameters['start_bounds'][6][0], fitting_parameters['start_bounds'][6][1])
                if fitting_parameters['set_age']:
                    if hull.find_simplex([m, a, np.log10(fitting_parameters['set_age'][0]*1e9)]) >= 0:
                        initial[count,4] = m
                        initial[count,6] = a
                        count += 1
                else:
                    A = np.random.uniform(fitting_parameters['start_bounds'][8][0], fitting_parameters['start_bounds'][8][1])
                    if hull.find_simplex([m, a, np.log10(A*1e9)]) >= 0:
                        initial[count,4] = m
                        initial[count,6] = a
                        initial[count,8] = A
                        count += 1
    
            count = 0
            # print('initializing params2')
            while count < fitting_parameters['walkers']:
                m = np.random.uniform(fitting_parameters['start_bounds'][5][0], fitting_parameters['start_bounds'][5][1])
                a = np.random.uniform(fitting_parameters['start_bounds'][7][0], fitting_parameters['start_bounds'][7][1])
                if fitting_parameters['set_age']:
                    if hull2.find_simplex([m, a, np.log10(fitting_parameters['set_age'][1]*1e9)]) >= 0:
                        initial[count,5] = m
                        initial[count,7] = a
                        count += 1
                else:
                    A = np.random.uniform(fitting_parameters['start_bounds'][9][0], fitting_parameters['start_bounds'][9][1])
                    if hull2.find_simplex([m, a, np.log10(A*1e9)]) >= 0:
                        initial[count,5] = m
                        initial[count,7] = a
                        initial[count,9] = A
                        count += 1
    
    if fit_type == 'two_populations':
        for ii in [0,1]:
            initial[:,ii] = np.random.uniform(fitting_parameters['start_bounds'][ii][0], fitting_parameters['start_bounds'][ii][1], fitting_parameters['walkers']) 
            
        count = 0
        while count < fitting_parameters['walkers']:
            m = np.random.uniform(fitting_parameters['start_bounds'][2][0], fitting_parameters['start_bounds'][2][1])
            a = np.random.uniform(fitting_parameters['start_bounds'][4][0], fitting_parameters['start_bounds'][4][1])
            if fitting_parameters['set_age']:
                if hull.find_simplex([m, a, np.log10(fitting_parameters['set_age'][0]*1e9)]) >= 0:
                    initial[count,2] = m
                    initial[count,4] = a
                    count += 1
            else:
                A = np.random.uniform(fitting_parameters['start_bounds'][6][0], fitting_parameters['start_bounds'][6][1])
                if hull.find_simplex([m, a, np.log10(A*1e9)]) >= 0:
                    initial[count,2] = m
                    initial[count,4] = a
                    initial[count,6] = A
                    count += 1

        count = 0
        while count < fitting_parameters['walkers']:
            m = np.random.uniform(fitting_parameters['start_bounds'][3][0], fitting_parameters['start_bounds'][3][1])
            a = np.random.uniform(fitting_parameters['start_bounds'][5][0], fitting_parameters['start_bounds'][5][1])
            if fitting_parameters['set_age']:
                if hull2.find_simplex([m, a, np.log10(fitting_parameters['set_age'][1]*1e9)]) >= 0:
                    initial[count,3] = m
                    initial[count,5] = a
                    count += 1
            else:
                A = np.random.uniform(fitting_parameters['start_bounds'][7][0], fitting_parameters['start_bounds'][7][1])
                if hull2.find_simplex([m, a, np.log10(A*1e9)]) >= 0:
                    initial[count,3] = m
                    initial[count,5] = a
                    initial[count,7] = A
                    count += 1


    return initial

###########################################################################
###########################################################################  
###########################################################################
"""This function evaluates the likelihood for each walker iteration in the 5D MCMC routine"""
###########################################################################

mu, sigma = 2, 0.5

def one_component_likelihood(start, Bprepped, Gprepped, Rprepped, bound, bin_width, template_params, age, disp_prior, use_hull):

    # Check if the parameters are within bounds
    if all(lower < val < upper for val, (lower, upper) in zip(start, bound)):

        in_cannon_training_set = True

        # Set template_params based on age and existing template_params
        if age:
            template_params = np.array([start[2], start[3], np.log10(age * 1e9)])
        elif not template_params:
            template_params = np.array([start[2], start[3], np.log10(start[4] * 1e9)])

        if use_hull:
            in_cannon_training_set = hull.find_simplex(template_params) >= 0

        if in_cannon_training_set:
            kinematics = [start[0], start[1]]
            log_likelihood = evaluate_onecomp_likelihood(kinematics, Bprepped, Gprepped, Rprepped, template_params, bin_width)

            # Return log_likelihood with or without prior
            if disp_prior:
                lnprior = (-1/2) * ((start[1] - mu) / sigma) ** 2
                return log_likelihood + lnprior
            return log_likelihood

    # If conditions are not met, return negative infinity
    return -np.inf

# def one_component_likelihood(start, Bprepped, Gprepped, Rprepped, bound, bin_width, template_params, age, disp_prior):
    
#     # Check if the parameters are within bounds
#     if all(lower <= val <= upper for val, (lower, upper) in zip(start, bound)):

#         # use this on the off chance you want to try correcting for the salaris metallicity. Which you shouldn't want to do.
#         # if age:
#         #     template_params = np.array([start[2]+np.log10((0.694*10**start[3])+0.306), start[3], np.log10(age*1e9)])
#         # elif not template_params:
#         #     template_params = np.array([start[2]+np.log10((0.694*10**start[3])+0.306), start[3], np.log10(start[4]*1e9)])
        
#         # Set template_params based on age and existing template_params
#         if age:
#             template_params = np.array([start[2], start[3], np.log10(age * 1e9)])
#         elif not template_params:
#             template_params = np.array([start[2], start[3], np.log10(start[4] * 1e9)])

#         # Check if template_params are within the convex hull
#         if model.in_convex_hull(template_params):
#             kinematics = [start[0], start[1]]
#             log_likelihood = evaluate_onecomp_likelihood(kinematics, Bprepped, Gprepped, Rprepped, template_params, bin_width)

#             # Return log_likelihood with or without prior
#             if disp_prior:
#                 lnprior = (-1/2) * ((start[1] - mu) / sigma) ** 2
#                 return log_likelihood + lnprior
#             return log_likelihood

#     # If conditions are not met, return negative infinity
#     return -np.inf



###########################################################################
###########################################################################
###########################################################################
"""Quick MCMC for best fitting single component result"""
###########################################################################
    
def one_component_fit(Bprepped, Gprepped, Rprepped, fitting_parameters, plot=True, save_plots_location=None):
    
    # initialize the first guesses for the start of the emcee as random variables pulled from a uniform distribution with start bounds
    

    if fitting_parameters['use_convex_hull']:
        initial = initialize_walkers_with_convex_hull(fitting_parameters, fit_type='one_component')

    else:
        initial = np.zeros((fitting_parameters['walkers'], fitting_parameters['dimensions']))
        for ii in range(fitting_parameters['dimensions']):
            initial[:,ii] = np.random.uniform(fitting_parameters['start_bounds'][ii][0], fitting_parameters['start_bounds'][ii][1], fitting_parameters['walkers'])

    

    # run the initial emcee routine
    with Pool(fitting_parameters['num_cpus']) as pool:
        sampler = emcee.EnsembleSampler(fitting_parameters['walkers'], fitting_parameters['dimensions'], one_component_likelihood, pool=pool, moves = emcee.moves.StretchMove(a=8), args=[Bprepped, Gprepped, Rprepped, fitting_parameters['fit_bounds'], fitting_parameters['bin_width'], fitting_parameters['set_params'], fitting_parameters['set_age'], fitting_parameters['use_disp_prior'], fitting_parameters['use_convex_hull']])

        sampler.run_mcmc(initial, fitting_parameters['iterations'], progress=True)
    
    samples = sampler.get_chain()
    lnprobs = sampler.get_log_prob()
    flat_samps = sampler.get_chain(flat=True, discard=fitting_parameters['burn_in'])
    flat_lnprb = sampler.get_log_prob(flat=True, discard=fitting_parameters['burn_in'])


    chain_results = []
    for ii in range(fitting_parameters['dimensions']):
        chain_results = np.hstack([chain_results, np.nanpercentile(flat_samps[flat_lnprb!=-np.inf,ii], [16,50,84])])
    maxL_results = flat_samps[np.argmax(flat_lnprb)]
    results = np.hstack([maxL_results, chain_results])

    plot_suffix = '_abunds.jpg'
         
    if fitting_parameters['set_params']:
        print('results from setting params')
        final_kins = np.copy(maxL_results)
        final_params = np.copy(fitting_parameters['set_params'])
        plot_suffix = '_kins.jpg'
        
    elif fitting_parameters['set_age']:
        print('results from setting age')
        final_kins = np.array([results[0], results[1]])
        final_params = np.array([results[2], results[3], np.log10(fitting_parameters['set_age']*1e9)])

    else:
        final_kins = np.array([results[0], results[1]])
        final_params = np.array([results[2], results[3], np.log10(results[4]*1e9)])
    
    blu_model, gre_model, red_model = generate_model(final_params, final_kins, fitting_parameters['bin_width'], Bprepped, Gprepped, Rprepped)

    # CALCULATE CHI SQUARED
    Bchi2 = evaluate_chi2(blu_model, Bprepped['gal'], Bprepped['pix'], Bprepped['err'])
    Gchi2 = evaluate_chi2(gre_model, Gprepped['gal'], Gprepped['pix'], Gprepped['err'])
    Rchi2 = evaluate_chi2(red_model, Rprepped['gal'], Rprepped['pix'], Rprepped['err'])

    chi2 = (Bchi2 + Gchi2 + Gchi2) / (len(Bprepped['pix']) + len(Gprepped['pix']) + len(Rprepped['pix']))
    
    if plot:
        plot_and_save.walker_chain(fitting_parameters['labels'], fitting_parameters['dimensions'], samples, lnprobs, save_plots_location, 'burn_in'+plot_suffix)
    
        plot_and_save.fit(Bprepped, Gprepped, Rprepped, blu_model, gre_model, red_model, save_plots_location, 'fit'+plot_suffix, plot_resids_here=0.5)
    
        plot_and_save.corner_plot(flat_samps, fitting_parameters['labels'], save_plots_location, 'corner'+plot_suffix)
    
    
    return results, chi2, blu_model, gre_model, red_model



###########################################################################
###########################################################################  
###########################################################################
"""This function evaluates the log likelihood value for each one-component model"""
###########################################################################

def evaluate_twocomp_likelihood(params1, params2, kinematics1, kinematics2, Bprepped, Gprepped, Rprepped, frac, bin_width):

    blu_model, gre_model, red_model = generate_twocomp_model(params1, params2, kinematics1, kinematics2, frac, bin_width, Bprepped, Gprepped, Rprepped)

    
    # CALCULATE CHI SQUARED
    Bchi2 = evaluate_chi2(blu_model, Bprepped['gal'], Bprepped['pix'], Bprepped['err'])
    Gchi2 = evaluate_chi2(gre_model, Gprepped['gal'], Gprepped['pix'], Gprepped['err'])
    Rchi2 = evaluate_chi2(red_model, Rprepped['gal'], Rprepped['pix'], Rprepped['err'])
    
    likelihood = -1*(Bchi2 + Gchi2 + Rchi2)/2

    return likelihood


###########################################################################
###########################################################################  
###########################################################################
"""This function uses the above function to generate a model to fit to the data for each chip"""
###########################################################################

def generate_twocomp_model(params1, params2, kinematics1, kinematics2, frac, bin_width, Bprepped, Gprepped, Rprepped):

    # generate the cannon templates
    test_design_matrix = model.vectorizer((params1 - model._fiducials)/model._scales)
    template1 = np.dot(model.theta, test_design_matrix)[:,0]  

    test_design_matrix = model2.vectorizer((params2 - model2._fiducials)/model2._scales)
    template2 = np.dot(model2.theta, test_design_matrix)[:,0]  

    # SPLIT CHIPS
    blu_temp1 = template1[Bprepped['lim'][0]:Bprepped['lim'][1]]
    gre_temp1 = template1[Gprepped['lim'][0]:Gprepped['lim'][1]]
    red_temp1 = template1[Rprepped['lim'][0]:Rprepped['lim'][1]]

    blu_temp2 = template2[Bprepped['lim'][0]:Bprepped['lim'][1]]
    gre_temp2 = template2[Gprepped['lim'][0]:Gprepped['lim'][1]]
    red_temp2 = template2[Rprepped['lim'][0]:Rprepped['lim'][1]]

    blu_temp1 = convolve_gauss_hermite(blu_temp1-1, Bprepped['vsc'], kinematics1, len(blu_temp1))+1
    gre_temp1 = convolve_gauss_hermite(gre_temp1-1, Gprepped['vsc'], kinematics1, len(gre_temp1))+1
    red_temp1 = convolve_gauss_hermite(red_temp1-1, Rprepped['vsc'], kinematics1, len(red_temp1))+1

    blu_temp2 = convolve_gauss_hermite(blu_temp2-1, Bprepped['vsc'], kinematics2, len(blu_temp2))+1
    gre_temp2 = convolve_gauss_hermite(gre_temp2-1, Gprepped['vsc'], kinematics2, len(gre_temp2))+1
    red_temp2 = convolve_gauss_hermite(red_temp2-1, Rprepped['vsc'], kinematics2, len(red_temp2))+1

    blu_comb_temp = (blu_temp1*frac) + (blu_temp2*(1-frac))
    gre_comb_temp = (gre_temp1*frac) + (gre_temp2*(1-frac))
    red_comb_temp = (red_temp1*frac) + (red_temp2*(1-frac))

    # TAKE bin_width PIXEL RUNNING MEDIAN CONTINUUM
    blu_model = blu_comb_temp/median_filter(blu_comb_temp, bin_width)
    gre_model = gre_comb_temp/median_filter(gre_comb_temp, bin_width)
    red_model = red_comb_temp/median_filter(red_comb_temp, bin_width)

    return blu_model, gre_model, red_model

###########################################################################
###########################################################################  
###########################################################################
"""This function evaluates the likelihood for each walker iteration in the 8D MCMC routine"""
###########################################################################

mu, sigma = 2, 0.5

def two_population_likelihood(start, Bprepped, Gprepped, Rprepped, bound, bin_width, template_params, ages, frac, disp_prior):
    
    # check if the parameters the MCMC guesses are outside our parameter space
    if all(lower < val < upper for val, (lower, upper) in zip(start, bound)):

        if ages:
            params1 = np.array([start[2], start[4], np.log10(ages[0]*1e9)])
            params2 = np.array([start[3], start[5], np.log10(ages[1]*1e9)])
            
        elif template_params:
            params1 = np.copy(template_params)
            params2 = np.array([start[2], start[3], np.log10(start[4]*1e9)])
            
        else:
            params1 = np.array([start[2], start[4], np.log10(start[6]*1e9)])
            params2 = np.array([start[3], start[5], np.log10(start[7]*1e9)])

        in_cannon_training_set = hull.find_simplex(params1) >= 0
        in_cannon2_training_set = hull2.find_simplex(params2) >= 0
            
        if not frac:
            frac = start[-1]

        if (in_cannon_training_set and in_cannon2_training_set and use_hull) | (not use_hull):
            
            log_likelihood = evaluate_twocomp_likelihood(params1, params2, [start[0], start[1]], [start[0], start[1]], Bprepped, Gprepped, Rprepped, frac, bin_width)
    
            if disp_prior:
                log_likelihood += (-1/2) * ((start[1]-mu)/sigma)**2
                return log_likelihood + lnprior
            return log_likelihood
        
    # if they are not, return negative infinity
    return -np.inf
    
    
###########################################################################
###########################################################################
###########################################################################
"""Quick MCMC for best fitting single component result"""
###########################################################################
    
def two_population_fit(Bprepped, Gprepped, Rprepped, fitting_parameters, plot=True, save_plots_location=None):
    
    # initialize the first guesses for the start of the emcee as random variables pulled from a uniform distribution with start bounds
    initial = np.zeros((fitting_parameters['walkers'], fitting_parameters['dimensions']))
    for ii in tqdm(range(fitting_parameters['dimensions'])):
        initial[:,ii] = np.random.uniform(fitting_parameters['start_bounds'][ii][0], fitting_parameters['start_bounds'][ii][1], fitting_parameters['walkers'])
        
    # run the emcee routine
    with Pool(fitting_parameters['num_cpus']) as pool:
        sampler = emcee.EnsembleSampler(fitting_parameters['walkers'], fitting_parameters['dimensions'], two_population_likelihood, pool=pool, moves = emcee.moves.StretchMove(a=8), args=[Bprepped, Gprepped, Rprepped, fitting_parameters['fit_bounds'], fitting_parameters['bin_width'], fitting_parameters['set_params'], fitting_parameters['set_age'], fitting_parameters['set_frac'], fitting_parameters['use_disp_prior']])
        
        sampler.run_mcmc(initial, fitting_parameters['iterations'], progress=True)
    
    samples = sampler.get_chain()
    lnprobs = sampler.get_log_prob()
    flat_samps = sampler.get_chain(flat=True, discard=fitting_parameters['burn_in'])
    flat_lnprb = sampler.get_log_prob(flat=True, discard=fitting_parameters['burn_in'])

    chain_results = []
    for ii in range(fitting_parameters['dimensions']):
        chain_results = np.hstack([chain_results, np.nanpercentile(flat_samps[flat_lnprb!=-np.inf,ii], [16,50,84])])
    maxL_results = flat_samps[np.argmax(flat_lnprb)]
    results = np.hstack([maxL_results, chain_results])

    plot_suffix = '_pops.jpg'
        
    if fitting_parameters['set_params']:
        print('results from setting params')
        final_params1 = np.copy(fitting_parameters['set_params'])
        final_params2 = np.array([results[2], results[3], np.log10(results[4]*1e9)])
        
    elif fitting_parameters['set_age']:
        print('results from setting age')
        final_params1 = np.array([results[2], results[4], np.log10(fitting_parameters['set_age'][0]*1e9)])
        final_params2 = np.array([results[3], results[5], np.log10(fitting_parameters['set_age'][1]*1e9)])

    else:
        final_params1 = np.array([results[2], results[4], np.log10(results[6]*1e9)])
        final_params2 = np.array([results[3], results[5], np.log10(results[7]*1e9)])

    if fitting_parameters['set_frac']:
        print('results from setting frac')
        final_frac = np.copy(fitting_parameters['set_frac'])
    else:
        final_frac = maxL_results[-1]

    final_kins = [results[0], results[1]]
        
    blu_model, gre_model, red_model = generate_twocomp_model(final_params1, final_params2, final_kins, final_kins, final_frac, fitting_parameters['bin_width'], Bprepped, Gprepped, Rprepped)

    # CALCULATE CHI SQUARED
    Bchi2 = evaluate_chi2(blu_model, Bprepped['gal'], Bprepped['pix'], Bprepped['err'])
    Gchi2 = evaluate_chi2(gre_model, Gprepped['gal'], Gprepped['pix'], Gprepped['err'])
    Rchi2 = evaluate_chi2(red_model, Rprepped['gal'], Rprepped['pix'], Rprepped['err'])

    chi2 = (Bchi2 + Gchi2 + Gchi2) / (len(Bprepped['pix']) + len(Gprepped['pix']) + len(Rprepped['pix']))
    
    if plot:
        plot_and_save.walker_chain(fitting_parameters['labels'], fitting_parameters['dimensions'], samples, lnprobs, save_plots_location, 'burn_in'+plot_suffix)
    
        plot_and_save.fit(Bprepped, Gprepped, Rprepped, blu_model, gre_model, red_model, save_plots_location, 'fit'+plot_suffix, plot_resids_here=0.5)
    
        plot_and_save.corner_plot(flat_samps, fitting_parameters['labels'], save_plots_location, 'corner'+plot_suffix)

    return results, chi2, blu_model, gre_model, red_model


###########################################################################
###########################################################################  
###########################################################################
"""This function evaluates the likelihood for each walker iteration in the 8D MCMC routine"""
###########################################################################

def two_component_likelihood(start, Bprepped, Gprepped, Rprepped, bound, bin_width, mean_velocity, template_params, ages, frac, use_hull, disp_prior):

    in_cannon_training_set, in_cannon2_training_set = True, True
    fractional_velocity = (frac**2)*start[0] + (1-frac**2)*start[1]
    
    # """ Check to make sure Comp 1 rotates slower and has higher dispersion than Comp 2 """
    if disp_prior == 'standard' and start[2] < start[3]:
        return -np.inf
    if disp_prior == 'reversed' and start[2] > start[3]:
        return -np.inf

    
    if np.abs(start[0] + 300) > np.abs(start[1] + 300) or np.sign(mean_velocity - start[0]) == np.sign(mean_velocity - start[1]):
        return -np.inf

    # """ Check to make sure the relational velocity equation (Eq. 5 of Gibson et al. 2025) is satisfied """
    if fractional_velocity <= (mean_velocity - mean_velocity_offset) or fractional_velocity >= (mean_velocity + mean_velocity_offset):
        return -np.inf
        
    # """ check if the parameters the MCMC guesses are outside our parameter space """
    if all(lower < val < upper for val, (lower, upper) in zip(start, bound)):

        if ages:
            params1 = np.array([start[4], start[6], np.log10(ages[0]*1e9)])
            params2 = np.array([start[5], start[7], np.log10(ages[1]*1e9)])
            
        elif template_params:
            params1 = np.copy(template_params)
            params2 = np.copy(template_params)
            
        else:
            params1 = np.array([start[4], start[6], np.log10(start[8]*1e9)])
            params2 = np.array([start[5], start[7], np.log10(start[9]*1e9)])

        if use_hull:
            in_cannon_training_set = hull.find_simplex(params1) >= 0
            in_cannon2_training_set = hull2.find_simplex(params2) >= 0

        if in_cannon_training_set and in_cannon2_training_set:
            
            if not frac:
                frac = start[-1]
                
            log_likelihood = evaluate_twocomp_likelihood(params1, params2, [start[0], start[2]], [start[1], start[3]], Bprepped, Gprepped, Rprepped, frac, bin_width)
        
            return log_likelihood
        
    # if they are not, return negative infinity
    return -np.inf


###########################################################################
###########################################################################
###########################################################################
"""Quick MCMC for best fitting single component result"""
###########################################################################
    
def two_component_fit(Bprepped, Gprepped, Rprepped, fitting_parameters, return_errors=False, plot=True, save_plots_location=None):

    initial = np.zeros((fitting_parameters['walkers'], fitting_parameters['dimensions']))

    if fitting_parameters['use_convex_hull']:
        initial = initialize_walkers_with_convex_hull(fitting_parameters, fit_type='two_components')
    else:
        count = 0
        while count < fitting_parameters['walkers']:
            v1 = np.random.uniform(fitting_parameters['start_bounds'][0][0], fitting_parameters['start_bounds'][0][1])
            v2 = np.random.uniform(fitting_parameters['start_bounds'][1][0], fitting_parameters['start_bounds'][1][1])
            vf = (fitting_parameters['set_frac']**2)*v1 + (1 - fitting_parameters['set_frac']**2)*v2
            if np.sign(fitting_parameters['vel'] - v1) != np.sign(fitting_parameters['vel'] - v2) and np.abs(v1 + 300) < np.abs(v2 + 300) and vf >= fitting_parameters['vel'] - mean_velocity_offset and vf <= fitting_parameters['vel'] + mean_velocity_offset:
                initial[count,0] = v1
                initial[count,1] = v2
                count += 1

        count = 0
        while count < fitting_parameters['walkers']:
            d1 = np.random.uniform(fitting_parameters['start_bounds'][2][0], fitting_parameters['start_bounds'][2][1])
            d2 = np.random.uniform(fitting_parameters['start_bounds'][3][0], fitting_parameters['start_bounds'][3][1])
            if fitting_parameters['disp_prior'] == 'standard' and d1 > d2 or fitting_parameters['disp_prior'] == 'reversed' and d1 < d2:
                initial[count,2] = d1
                initial[count,3] = d2
                count += 1

        if not fitting_parameters['set_params']:
            for ii in range(4, fitting_parameters['dimensions']):
                initial[:,ii] = np.random.uniform(fitting_parameters['start_bounds'][ii][0], fitting_parameters['start_bounds'][ii][1], fitting_parameters['walkers'])

    

    # run the emcee routine
    with Pool(fitting_parameters['num_cpus']) as pool:
        sampler = emcee.EnsembleSampler(fitting_parameters['walkers'], fitting_parameters['dimensions'], two_component_likelihood, pool=pool, moves = emcee.moves.StretchMove(a=8), args=[Bprepped, Gprepped, Rprepped, fitting_parameters['fit_bounds'], fitting_parameters['bin_width'], fitting_parameters['vel'], fitting_parameters['set_params'], fitting_parameters['set_age'], fitting_parameters['set_frac'], fitting_parameters['use_convex_hull'], fitting_parameters['disp_prior']])
        
        sampler.run_mcmc(initial, fitting_parameters['iterations'], progress=True)
    
    samples = sampler.get_chain()
    lnprobs = sampler.get_log_prob()
    flat_samps = sampler.get_chain(flat=True, discard=fitting_parameters['burn_in'])
    flat_lnprb = sampler.get_log_prob(flat=True, discard=fitting_parameters['burn_in'])

    chain_results = []
    for ii in range(fitting_parameters['dimensions']):
        chain_results = np.hstack([chain_results, np.nanpercentile(flat_samps[flat_lnprb!=-np.inf,ii], [16,50,84])])
    maxL_results = flat_samps[np.argmax(flat_lnprb)]
    results = np.hstack([maxL_results, chain_results])

    plot_suffix = '_twocomp.jpg'
    
    if fitting_parameters['set_params']:
        print('results from setting params')
        final_params1 = np.copy(fitting_parameters['set_params'])
        final_params2 = np.copy(fitting_parameters['set_params'])
        
    elif fitting_parameters['set_age']:
        print('results from setting age')
        final_params1 = np.array([results[4], results[6], np.log10(fitting_parameters['set_age'][0]*1e9)])
        final_params2 = np.array([results[5], results[7], np.log10(fitting_parameters['set_age'][1]*1e9)])

    else:
        final_params1 = np.array([results[4], results[6], np.log10(results[8]*1e9)])
        final_params2 = np.array([results[5], results[7], np.log10(results[9]*1e9)])

    if fitting_parameters['set_frac']:
        print('results from setting frac')
        final_frac = np.copy(fitting_parameters['set_frac'])
    else:
        final_frac = maxL_results[-1]

    final_kins1 = [results[0], results[2]]
    final_kins2 = [results[1], results[3]]

    blu_model, gre_model, red_model = generate_twocomp_model(final_params1, final_params2, final_kins1, final_kins2, final_frac, fitting_parameters['bin_width'], Bprepped, Gprepped, Rprepped)

    # CALCULATE CHI SQUARED
    Bchi2 = evaluate_chi2(blu_model, Bprepped['gal'], Bprepped['pix'], Bprepped['err'])
    Gchi2 = evaluate_chi2(gre_model, Gprepped['gal'], Gprepped['pix'], Gprepped['err'])
    Rchi2 = evaluate_chi2(red_model, Rprepped['gal'], Rprepped['pix'], Rprepped['err'])

    chi2 = (Bchi2 + Gchi2 + Gchi2) / (len(Bprepped['pix']) + len(Gprepped['pix']) + len(Rprepped['pix']))
    
    if plot:
        plot_and_save.walker_chain(fitting_parameters['labels'], fitting_parameters['dimensions'], samples, lnprobs, save_plots_location, 'burn_in'+plot_suffix)
    
        plot_and_save.fit(Bprepped, Gprepped, Rprepped, blu_model, gre_model, red_model, save_plots_location, 'fit'+plot_suffix, plot_resids_here=0.5)
    
        plot_and_save.corner_plot(flat_samps, fitting_parameters['labels'], save_plots_location, 'corner'+plot_suffix)

    return results, chi2, blu_model, gre_model, red_model

