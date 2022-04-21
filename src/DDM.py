import numba
import numpy as np
import cupy as cp
from numba import cuda, prange
from numba.cuda import random as cuda_random
from numba import jit

@jit
def DDM_cpu_sim(stimulus, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time): 
    """ 
    CPU version of model with drift-diffusion model with urgecy-gating and within-trial dynamic stimulus, boundary and drift variability.
    Additionally, implementation of starting point offset and drift rate offset for bias terms.
    
    Args:
        stimulus: Stimulus intensities with shape (num_trials, num_choices, num, time). Type: array(3D, float32)
        starting_point: 1D array of starting point for each accumulator with length equal to number of time steps. Type: array(1D, float32) 
        drift_gain: Common drift rate of accumulators (same for all accumulators). Type: float32
        drift_variability: Time-dependent drift-variability variability for each time point with length equal to number of time steps. Type array(1D, float32)
        drift_offset: Offset in drift-rate i.e., drift-rate in case of no/random stimulus with length equal to number of choices. Type: array(1D, float32) 
        decision_bound: Time dependent decision-bound. Array of length equal to number of time steps. Type: array(1D, float32)
        nondecision_time: Non-decision time. Type: float32
        urgency_signal: Boolean for linerly time-dependent urgency signal. Type: bool
        rng_states: random normal generator for cuda. Must have same number of states as parellelization (i.e., num_trials)
        
    Return:
        decision: Decision on each trial as a number of accumulator (0:num_choices-1) with length equal to number of trials. Type: array(1D, float32)
        reaction_time: Reaction time on each trial with length equal to number of trials. Type: array(1D, float32)       
    
        
    Raises:
        TypeError: data must be a list or numpy array
        ValueError: data must be m by 3 matrix.

    Information:
        2022-03 VT wrote it
        
    """
    
    if stimulus.ndim < 2 or stimulus.ndim > 3:        
        raise ValueError('Stimulus must be 2 or 3 dimensional array. \
                         3-dimensional - (num_trials, num_choices, num_samples) \
                         2-dimensional - (num_choices, num_samples)'
                        )
        
    if drift_variability.shape[0] != stimulus.shape[2]:        
        raise ValueError('Must provide drift variability as array with length eqaul to num_samples')
        
    if decision_bound.shape[0] != stimulus.shape[2]:        
        raise ValueError('Must provide decision bound as array with length eqaul to num_samples')
        
    decision = np.empty(stimulus.shape[0])*np.NaN
    reaction_time = np.empty(stimulus.shape[0])*np.NaN    
    
    # Loop over num_trials
    for tr in prange(stimulus.shape[0]):
        decision_variable = starting_point
        # Loop over num_samples
        for t in range(stimulus.shape[2]):
            diffusion_step = ((stimulus[tr,0,t]-stimulus[tr,1,t]) * drift_gain) + drift_offset + (np.random.normal(0,1)*drift_variability[t])
            decision_variable += diffusion_step # update decision variable
            if decision_variable > decision_bound[t] or decision_variable < -decision_bound[t]:       # checking if decision bound is reached
                decision[tr] =  2*(decision_variable>0) - 1    # np.sign(dv) alternative
                reaction_time[tr] = t+nondecision_time
                break  


    return decision, reaction_time



@cuda.jit
def DDM_kernel(stimulus, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time, urgency_signal, decision, reaction_time, rng_states): 
    """ 
    Cuda kernel for model with drift-diffusion model with uegecy-gating and within-trial dynamic stimulus, boundary and drift variability.
    Additionally, implementation of starting point offset and drift rate offset for bias terms.
    
    Note: Since cuda kernels cannot return values, we pass output variables to in function as mutable variables.
    
    Args:
        stimulus: Stimulus intensities with shape (num_trials, num_choices, num, time). Type: array(3D, float32)
        starting_point: 1D array of starting point for each accumulator with length equal to number of time steps. Type: array(1D, float32) 
        drift_gain: Common drift rate of accumulators (same for all accumulators). Type: float32
        drift_variability: Time-dependent drift-variability variability for each time point with length equal to number of time steps. Type array(1D, float32)
        drift_offset: Offset in drift-rate i.e., drift-rate in case of no/random stimulus with length equal to number of choices. Type: array(1D, float32) 
        decision_bound: Time dependent decision-bound. Array of length equal to number of time steps. Type: array(1D, float32)
        nondecision_time: Non-decision time. Type: float32
        urgency_signal: Boolean for linerly time-dependent urgency signal. Type: bool
        rng_states: random normal generator for cuda. Must have same number of states as parellelization (i.e., num_trials)
        
    Return:
        decision: Decision on each trial as a number of accumulator (0:num_choices-1) with length equal to number of trials. Type: array(1D, float32)
        reaction_time: Reaction time on each trial with length equal to number of trials. Type: array(1D, float32)       
    
        
    Raises:
        TypeError: data must be a list or numpy array
        ValueError: data must be m by 3 matrix.

    Information:
        2022-03 VT wrote it
        
    """
    
    # Input grid and stride size
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    # Loop over number of trials (parellel)
    for tr in range(start, stimulus.shape[0], stride): 
        decision_variable = starting_point     
        
        # Loop over number of samples
        for t in range(stimulus.shape[2]):
            diffusion_step = ((stimulus[tr,0,t]-stimulus[tr,1,t]) * drift_gain) + (cuda_random.xoroshiro128p_normal_float32(rng_states, tr)*drift_variability[t])   
            decision_variable += diffusion_step + drift_offset      # update decision variable
            if urgency_signal:
                decision_variable *= t
                
            # Check if bound reached
            if decision_variable > decision_bound[t] or decision_variable < -decision_bound[t]:       # checking if decision bound is reached
                decision[tr] =  2*(decision_variable>0) - 1    # np.sign(dv) alternative
                reaction_time[tr] = t+nondecision_time
                break  


def DDM_gpu_sim(stimulus, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time, urgency_signal, blockdim=128, batch_size=None, seed=None):
    """
    Batch simulation for individual_accumulator kernel. If the stimulus size is too big, GPU memory might overload hence splitting the data in multiple batches
    
    Args:
        stimulus: Stimulus intensities with shape (num_trials, num_choices, num, time). Type: array(3D, float32)
        starting_point: 1D array of starting point for each accumulator with length equal to number of time steps. Type: array(1D, float32) 
        drift_gain: Common drift rate of accumulators (same for all accumulators). Type: float32
        drift_variability: Time-dependent drift-variability variability for each time point with length equal to number of time steps. Type array(1D, float32)
        drift_offset: Offset in drift-rate i.e., drift-rate in case of no/random stimulus with length equal to number of choices. Type: array(1D, float32) 
        decision_bound: Time dependent decision-bound. Array of length equal to number of time steps. Type: array(1D, float32)
        nondecision_time: Non-decision time. Type: float32
        urgency_signal: Boolean for linerly time-dependent urgency signal. Type: bool
        batch_size: Batch size for simulation. Default value is 10000
        seed: Seed for randon number generator for diffusion process. Seed is set randomly is value is not provided
        
    Return:
        decision: Decision on each trial as a number of accumulator (0:num_choices-1) with length equal to number of trials. Type: array(1D, float32)
        reaction_time: Reaction time on each trial with length equal to number of trials. Type: array(1D, float32)       
    
        
    Raises:
        TypeError: data must be a list or numpy array
        ValueError: data must be m by 3 matrix.

    Information:
        2022-03 VT write it
    
    """
    
    # Input validation
    import warnings    
    
    if stimulus.ndim < 2 or stimulus.ndim > 3:        
        raise ValueError('Stimulus must be 2 or 3 dimensional array. \
                         3-dimensional - (num_trials, num_choices, num_samples) \
                         2-dimensional - (num_choices, num_samples)'
                        )
                   
    if drift_variability.shape[0] != stimulus.shape[2]:        
        raise ValueError('Must provide drift variability as array with length eqaul to num_samples')
                
    if decision_bound.shape[0] != stimulus.shape[2]:        
        raise ValueError('Must provide decision bound as array with length eqaul to num_samples')

    # If batch_size not provided, simulate whole data
    if batch_size is None:
        batch_size = stimulus.shape[0]    


    # Setting up parellel grid
    blockdim = 128
    griddim = (batch_size // blockdim) + 1

    # Setting random seed if not provided.
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng_states = cuda_random.create_xoroshiro128p_states(int(np.prod(blockdim) * np.prod(griddim)), seed)

    global num_choices
    num_choices = stimulus.shape[1]
    decision_np = []
    reaction_time_np = []

    # Batch simulation
    for i in range((stimulus.shape[0]//batch_size)+1):
        stimulus_batch = cp.asarray(stimulus[batch_size*i:batch_size*(i+1)])
        decision_cp = cp.empty(stimulus_batch.shape[0])*cp.NaN
        reaction_time_cp = cp.empty(stimulus_batch.shape[0])*cp.NaN
        
        cuda.synchronize()
        DDM_kernel[griddim, blockdim](stimulus_batch, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time, urgency_signal, decision_cp, reaction_time_cp, rng_states)
        cuda.synchronize()
        
        decision_np = np.append(decision_np, cp.asnumpy(decision_cp))
        reaction_time_np = np.append(reaction_time_np, cp.asnumpy(reaction_time_cp))
        
    return decision_np, reaction_time_np



def initialize_DDM_kernel(num_trials=3, num_choices=2, num_samples=1000):
    
    """
    Initializing GPU kernels for DDM model if GPU is available
    
    Args:
        gpu_available: Boolean
        
    Return:
        None
    """

    coherence = np.ones((num_trials, num_samples))*36     # 100

    starting_point = 0 #np.array(np.zeros(1), dtype=float32)             
    drift_offset = 0 #np.array(np.zeros(1), dtype=float32)
    drift_gain = float32(5e-5)             # drift gain
    drift_variability = float32(10e-3)      # diffusion variability
    nondecision_time = float32(100)         # Non-decision time (msec)
    decision_bound = 1
    bound_rate = 0
    bound_delay = 0
    urgency_signal = False
    # Dynamic time-dependent variables
    stimulus_cp = cp.asarray(get_unsigned_coherence_matrix(coherence))
    decision_bound = cp.asnumpy(get_time_dependent_bound(decision_bound, bound_rate, bound_delay))
    drift_variability = cp.asnumpy(get_time_dependent_variability(drift_variability))

    decision = cp.empty(stimulus_cp.shape[0])*cp.NaN
    reaction_time = cp.empty(stimulus_cp.shape[0])*cp.NaN

    blockdim = (256)
    griddim = stimulus_cp.shape[0] // (blockdim) + 1
    rng_states = cuda_random.create_xoroshiro128p_states(int(np.prod(blockdim) * np.prod(griddim)), seed=3)

    cuda.synchronize()
    DDM_kernel[griddim, blockdim](stimulus_cp, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time, urgency_signal, decision, reaction_time, rng_states)
    cuda.synchronize()    


