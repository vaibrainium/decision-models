import numba
import numpy as np
import cupy as cp
from numba import cuda, prange
from numba.cuda import random as cuda_random
from numba import jit, njit, vectorize
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)



def ReLU(input):
    '''Takes an input value and performs rectified linear unit operation i.e., 
    A = A   if A >= 0
    A = 0   if A <  0
    '''
    return input * (input >= 0)


def get_unsigned_coherence_matrix(normalized_signed_stimulus, num_choices = 2):
    '''
    Most of the decision-making studies use 2AFC task where usually choices are signed as positive for one choice and negative for another choice.
    Through this function we seperate this nomenclature by oupting two unsigned inputs rather than one signed input.
    
    Args: 
        normalized_signed_stimulus: signed stimulus difficulty e.g., coherence with num_trials x stop_time dimensions
    
    return:
        normalized_unsigned_stimulus: For 2-AFC task, returns stimulus matrix with dimensions num_trials x num_choices x stop_time
    '''
    
    # Input validation
    if not isinstance(normalized_signed_stimulus, np.ndarray):
        raise TypeError('Input must be a 1D or 2D numpy array with num_trials x stop_time dimensions')
        
    if normalized_signed_stimulus.ndim > 2:
        raise TypeError('Input must be a 1D or 2D numpy array with num_trials x stop_time dimensions')
                
    if normalized_signed_stimulus.ndim == 1:
        if len(normalized_signed_stimulus)<2:
            raise TypeError('Input must be a 1D or 2D numpy array with num_trials x stop_time dimensions with minumum 2 time points')
        else:
            normalized_unsigned_stimulus = np.zeros((num_choices, normalized_signed_stimulus.shape[0]), dtype=np.float32)
            normalized_unsigned_stimulus[0,:] = normalized_signed_stimulus * (normalized_signed_stimulus>0)   # If stimulus was positive keep it same for first cell else zero
            normalized_unsigned_stimulus[1,:] = -normalized_signed_stimulus * (normalized_signed_stimulus<0)   # If stimulus was negative take it's negative for second cell else zero
    
    if normalized_signed_stimulus.ndim == 2:
        normalized_unsigned_stimulus = np.zeros((normalized_signed_stimulus.shape[0], num_choices, normalized_signed_stimulus.shape[1]), dtype=np.float32)
        normalized_unsigned_stimulus[:,0,:] = normalized_signed_stimulus * (normalized_signed_stimulus>0)   # If stimulus was positive keep it same for first cell else zero
        normalized_unsigned_stimulus[:,1,:] = -normalized_signed_stimulus * (normalized_signed_stimulus<0)   # If stimulus was negative take it's negative for second cell else zero
 
    normalized_unsigned_stimulus = np.abs(normalized_unsigned_stimulus)
    
    return normalized_unsigned_stimulus



def get_time_dependent_bound(initial_bound, rate, delay, stop_time=10000):
    '''
    Generates time-dependent bound such as collapsing bound array
    args:
        initial_bound: Initial height of bounds
        rate: Rate of exponential decay. Negative means decreasing bound and +ve means increasings bounds
        delay: What point should decay start
        stop_time: To determine length of array
        
    returns:
        time_dependent_bound: 1D array with either exponential decay or rise
    '''
    
    time_dependent_bound = cp.ones(stop_time)*initial_bound
    
    return time_dependent_bound.astype(cp.float32)
    

def get_time_dependent_variability(initial_variability, time_coefficient=0, stop_time=10000):
    '''
    Generates linear time-dependent sigma either increasing, decreasing or constant
    args:
        initial_variability: Initial value of diffusion variability (sigma)
        time_coefficient: Rate of linear time-dependency. Negative means decreasing bound and +ve means increasings diffusion variability. Default value 0 i.e., constant variability
        stop_time: To determine length of array (Default value 10000 or 10 seconds)
        
    returns:
        time_dependent_variability: 1D array with either constant or linearly time-dependent diffusion variability
    '''
    
    time_dependent_variability = (cp.ones(stop_time)*initial_variability) + (time_coefficient*cp.arange(stop_time))
    
    return time_dependent_variability.astype(cp.float32)