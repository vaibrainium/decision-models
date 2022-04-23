
import numpy as np
from KDEpy import FFTKDE  # Fastest 1D algorithm



def get_llh_PDE(RT_model, RT_data, kernel='gaussian', bandwidth='silverman'):
    """
    Takes reaction time distribution of model and data and calculated negative log likelihood based on the Probability Density Estimation.
    
    Args: 
        RT_model:  Reaction time data from model simulation in msec. Type: array(1D, float32)
        RT_data:   Reaction time data from experimental data in msec. Type: array(1D, float32)
        kernal:    Choose kernel for PDE estimation. Available options:
                   ['gaussian', 'exponential', 'box', 'tri', 'epa', 'biweight', 'triweight', 'tricube', 'cosine']
                   Default: 'gaussian' kernel
        bandwidth: 
        
    Return:
        nLLh: Negatively log likelihood of the data given model parameters
    """
 
    # if "FFTKDE" not in dir():
    #     raise ValueError('You have not imported FFTKDE from KDEpy')
        
    if kernel not in FFTKDE._available_kernels.keys():
        raise ValueError('Specified kernel not in package. Please use one of following kernels: \n [\'gaussian\', \'exponential\', \'box\', \'tri\', \'epa\', \'biweight\', \'triweight\', \'tricube\', \'cosine\']')
    
    if not str(bandwidth).isnumeric() and (bandwidth not in FFTKDE._bw_methods.keys()):
        raise ValueError('Specified bandwidth not applicable. Please either use number of automatically choose with one of the following method: \n [\'silverman\', \'scott\', \'ISJ\'] ')
    
    
    RT_range = np.arange(np.max(np.concatenate((RT_data,RT_model))) + 1)
    PDE_model = FFTKDE(kernel=kernel, bw=bandwidth).fit(RT_model).evaluate(RT_range)
    PDE_data = FFTKDE(kernel='gaussian', bw='silverman').fit(RT_data).evaluate(RT_range)
    N_data = PDE_data * len(RT_data)  # Converting probability to total number of trials  

    likelihood = (PDE_model ** N_data) + 1e-24   # Calculating likelihood and offseting by small number to avoid calculation of log(0)
    nLLh = sum(-np.log(likelihood))    
    return nLLh



def get_llh_QMLE(RT_model, RT_data, nbins=9):
	"""
	Takes reaction time distribution of model and data and calculated negative log likelihood based on the Quasi-Maximum Likelihood Estimate (QMLE).

	Args: 
	    RT_model: Reaction time data from model simulation in msec. Type: array(1D, float32)
	    RT_data: Reaction time data from experimental data in msec. Type: array(1D, float32)
	    nbins: Quantized bins. Default value is 9.
	    
	Return:
	    nLLh: Negatively log likelihood of the data given model parameters
	    
	"""

	if nbins > len(RT_model):        
	    raise ValueError('Number of bins cannot be more than number samples')
	     
	quantile = np.arange(0, 1+1e-10, 1/nbins)
	nTrials = len(RT_data)
	trials_per_quantile = np.diff(quantile) * nTrials
	quantile_estimate = np.zeros(nbins+1)*np.nan
	QML = np.zeros(nbins)*np.nan

	quantile_estimate[0] = min(RT_data)
	quantile_estimate[-1] = max(RT_data)

	for bin_num in range(1,nbins):
	    centre = (quantile[bin_num]*nTrials) + 0.5
	    prev_edge = int(np.floor(centre))
	    next_edge = int(np.ceil(centre))
	    quantile_estimate[bin_num] = RT_data[prev_edge-1] + (RT_data[next_edge-1] - RT_data[prev_edge-1]) * (centre - prev_edge)

	for bin_num in range(0,nbins):
	    likelihood = np.mean(np.logical_and(RT_model>=quantile_estimate[bin_num], RT_model<quantile_estimate[bin_num+1])) ** trials_per_quantile[bin_num]
	    QML[bin_num] = -np.log(likelihood + 1e-24)   # Offseting likelihood by small number to avoid calculation of log(0)   
	nLLh = sum(QML)

	return nLLh