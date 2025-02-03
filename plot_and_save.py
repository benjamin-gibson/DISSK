#!usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import thecannon as tc
from tqdm import tqdm
import corner
from scipy.ndimage import median_filter


###########################################################################
"""Import all the necessary files and initialize directories"""
###########################################################################

homedir = '/uufs/astro.utah.edu/common/home/u1267339/'
alist_dir = '/uufs/astro.utah.edu/common/home/u1267339/ALIST/'

apogee_wave = np.loadtxt(homedir + 'InputData/lambdatemp.txt')

###########################################################################
###########################################################################
###########################################################################
"""this function plots the walker chain of the MCMC"""
###########################################################################

def walker_chain(labels, dimensions, samples, lnprobs, save_plots_location, fig_name):

    fig_height = int(4*(dimensions+1))
    plt.figure(figsize=(12,fig_height))
    for i in range(dimensions):
        plt.subplot(dimensions+1,1,i+1)
        plt.plot(samples[:, :, i])
        plt.ylabel(labels[i])
        plt.gca().set_xticklabels([])
    plt.subplot(dimensions+1,1,dimensions+1)
    plt.plot(lnprobs)
    plt.ylim(np.nanmin(lnprobs[-1,:]), np.nanmax(lnprobs[-1,:]))
    plt.ylabel('Likelihood')

    plt.subplots_adjust(hspace=0, wspace=0)
    
    if not save_plots_location:
        plt.show()
    else:
        plt.savefig(save_plots_location + fig_name, bbox_inches='tight', dpi=200)
        plt.close()



def fit(Bprepped, Gprepped, Rprepped, blu_model, gre_model, red_model, save_plots_location, fig_name, plot_resids_here=0.5):

        plt.figure(figsize=(15,17))
        plt.subplot(3,1,1)
        plt.plot(Bprepped['wav'], Bprepped['gal'], c='k', zorder=2)
        plt.plot(Bprepped['wav'], blu_model, c='r', zorder=3)
        plt.vlines(Bprepped['wav'][Bprepped['bad']], 0, 2, color='lightgray', zorder=1)
        plt.hlines(plot_resids_here, Bprepped['wav'][0], Bprepped['wav'][-1], color='k', linestyle='dashed')
        plt.scatter(Bprepped['wav'][Bprepped['pix']], (Bprepped['gal']-blu_model)[Bprepped['pix']] + plot_resids_here, c='lime', marker='.', zorder=0)
        plt.ylim(.3, 1.2)
        plt.xlim(Bprepped['wav'][0], Bprepped['wav'][-1])
#         plt.ylim(np.min([.7, np.min(Bprepped['gal'][Bprepped['pix']])]), np.max([1.1, np.max(Bprepped['gal'][Bprepped['pix']])]))


        plt.subplot(3,1,2)
        plt.plot(Gprepped['wav'], Gprepped['gal'], c='k', zorder=2)
        plt.plot(Gprepped['wav'], gre_model, c='r', zorder=3)
        plt.vlines(Gprepped['wav'][Gprepped['bad']], 0, 2, color='lightgray', zorder=1)
        plt.hlines(plot_resids_here, Gprepped['wav'][0], Gprepped['wav'][-1], color='k', linestyle='dashed')
        plt.scatter(Gprepped['wav'][Gprepped['pix']], (Gprepped['gal']-gre_model)[Gprepped['pix']] + plot_resids_here, c='lime', marker='.', zorder=0)
        plt.ylim(.3, 1.2)
        plt.xlim(Gprepped['wav'][0], Gprepped['wav'][-1])
#         plt.ylim(np.min([.7, np.min(Gprepped['gal'][Gprepped['pix']])]), np.max([1.1, np.max(Gprepped['gal'][Gprepped['pix']])]))

        plt.subplot(3,1,3)
        plt.plot(Rprepped['wav'], Rprepped['gal'], c='k', zorder=2)
        plt.plot(Rprepped['wav'], red_model, c='r', zorder=3)
        plt.vlines(Rprepped['wav'][Rprepped['bad']], 0, 2, color='lightgray', zorder=1)
        plt.hlines(plot_resids_here, Rprepped['wav'][0], Rprepped['wav'][-1], color='k', linestyle='dashed')
        plt.scatter(Rprepped['wav'][Rprepped['pix']], (Rprepped['gal']-red_model)[Rprepped['pix']] + plot_resids_here, c='lime', marker='.', zorder=0)
        plt.ylim(.3, 1.2)
        plt.xlim(Rprepped['wav'][0], Rprepped['wav'][-1])
#         plt.ylim(np.min([., np.min(Rprepped['gal'][Rprepped['pix']])]), np.max([1.1, np.max(Rprepped['gal'][Rprepped['pix']])]))

        plt.subplots_adjust(hspace=0.1)

        if not save_plots_location:
            plt.show()
        else:
            plt.savefig(save_plots_location + fig_name, bbox_inches='tight', dpi=200)
            plt.close()




def corner_plot(flat_samples, labels, save_plots_location, fig_name):

        figure = corner.corner(flat_samples, labels=labels, 
                               quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 15})
        for ax in figure.get_axes():
            ax.tick_params(axis='both', labelsize=12)
            
        if not save_plots_location:
            plt.show()
        else:
            plt.savefig(save_plots_location + fig_name, bbox_inches='tight', dpi=100)
            plt.close()