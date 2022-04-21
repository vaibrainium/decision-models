import numba
import numpy as np
import cupy as cp
from numba import cuda, prange
from numba.cuda import random as cuda_random
from numba import jit, njit, vectorize
from numba.core.errors import NumbaPerformanceWarning



@jit        
def IAM_cpu_sim(stimulus, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time, lateral_inhibition, leak, neural_ddm, urgency_signal): 
    """ CPU version of model with drift, leak, lateral inhibition (competiting), ddm like stimulus driven inhibition and urgency model. 
    Also works with within-trial dynamic stimulus, boundary and drift variability. Additionally, implementation of starting point offset and drift rate offset for bias terms.
    
    Note: Since cuda kernels cannot return values, we pass output variables to in function as mutable variables.
    
    Args:
        stimulus: Stimulus intensities with shape (num_trials, num_choices, num, time). Type: array(3D, float32)
        starting_point: 1D array of starting point for each accumulator with length equal to number of time steps. Type: array(1D, float32) 
        drift_gain: Common drift rate of accumulators (same for all accumulators). Type: float32
        drift_variability: Time-dependent drift-variability variability for each time point with length equal to number of time steps. Type array(1D, float32)
        drift_offset: Offset in drift-rate i.e., drift-rate in case of no/random stimulus with length equal to number of choices. Type: array(1D, float32) 
        decision_bound: Time dependent decision-bound. Array of length equal to number of time steps. Type: array(1D, float32)
        nondecision_time: Non-decision time. Type: float32
        lateral_inhibition: Waight of lateral inhibition from other accumulator. Type: float32
        leak: Leak factor from indivdual accumulator. Proportional to evidence in respective accumulator. Type: float32
        neural_ddm: DDM like lateral inhibition from drift-rate from other stimulus intensities. Type: bool
        urgency_signal: Boolean for linerly time-dependent urgency signal. Type: bool
        
    Return:
        decision: Decision on each trial as a number of accumulator (0:num_choices-1) with length equal to number of trials. Type: array(1D, float32)
        reaction_time: Reaction time on each trial with length equal to number of trials. Type: array(1D, float32)       
    
        
    Raises:
        TypeError: data must be a list or numpy array
        ValueError: data must be m by 3 matrix.

    Information:
        2022-03 VT write it
        
    """
    
    
    if stimulus.ndim < 2 or stimulus.ndim > 3:        
        raise ValueError('Stimulus must be 2 or 3 dimensional array. \
                         3-dimensional - (num_trials, num_choices, num_samples) \
                         2-dimensional - (num_choices, num_samples)'
                        )
           
    if starting_point.shape[0] != stimulus.shape[1]:        
        raise ValueError('Must provide starting point as array with length eqaul to num_choices')
        
    if drift_variability.shape[0] != stimulus.shape[2]:        
        raise ValueError('Must provide drift variability as array with length eqaul to num_samples')
        
    if drift_offset.shape[0] != stimulus.shape[1]:        
        raise ValueError('Must provide drift offset as array with length eqaul to num_choices')
        
    if decision_bound.shape[0] != stimulus.shape[2]:        
        raise ValueError('Must provide decision bound as array with length eqaul to num_samples')
        
    decision = np.empty(stimulus.shape[0])*cp.NaN
    reaction_time = np.empty(stimulus.shape[0])*cp.NaN    

    # Loop over num_trials
    for tr in prange(stimulus.shape[0]):
        decision_variable = starting_point.copy()      
        sum_decision_variables = decision_variable.sum()
        boundary_reached = False
            
        # Loop over num_samples
        for t in range(stimulus.shape[2]):            
            reached = 0
            sum_decision_variables = decision_variable.sum()
            drift_rate = stimulus[tr,:,t] * drift_gain
            
            # Loop over num_choices (i.e., accumulators)
            for accumulator in range(stimulus.shape[1]):
                
                diffusion_step = drift_rate[accumulator] + drift_offset[accumulator] + (np.random.normal(0,1)*drift_variability[t])  
                leak_step = leak * decision_variable[accumulator]
                lateral_dv_inhibition_step = lateral_inhibition * (sum_decision_variables - decision_variable[accumulator])  # Lateral inhibition from all decision_variables except self
                ddm_like_dr_inhibition_step = drift_rate.sum() - drift_rate[accumulator] + drift_offset.sum() - drift_offset[accumulator]
                
                decision_step = decision_variable[accumulator] \
                                + diffusion_step \
                                - leak_step \
                                - lateral_dv_inhibition_step \
                                - ddm_like_dr_inhibition_step*neural_ddm
                
                if urgency_signal == True:
                    decision_step*=t
                
                decision_variable[accumulator] = decision_step * (decision_step > 0)
                
                if decision_variable[accumulator] > decision_bound[t]:
                    boundary_reached = True
                    decision[tr] = accumulator
                    reaction_time[tr] = t + nondecision_time
                    break
                    
            if boundary_reached == True:
                break        
            
    return decision, reaction_time




@cuda.jit(device=True)
def ReLU_cuda(input):
    '''Takes an input value and performs rectified linear unit operation i.e., 
    A = A   if A >= 0
    A = 0   if A <  0
    Args:
        input 
    Return:
        ReLU(input)   
    '''
    return input * (input >= 0)


@cuda.jit(device=True)
def summation_cuda(matrix):
    """
    Cuda kernel for taking sum of all elements in a matrix  
    Args:
        vector: nD array
    Return:
        summation: Summation of all elements in a given matrix           
    """   
        
    summation = 0
    for _, val in enumerate(matrix.flat):
        summation += val
    return summation    
    
    
    
@cuda.jit
def vector_scaling_cuda(vector, scalar, scaled_vector):
    """
    Cuda kernel for scaling a vector with a scalar 
    Args:
        vector: 1D array
        scalar: scaling value
    Return:
        scaled_vector: scalar multiplication of given vector        
    """   
    
    for i, val in enumerate(vector):
        scaled_vector[i] = val*scalar



@cuda.jit
def IAM_kernel(stimulus, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time, lateral_inhibition, leak, neural_ddm, urgency_signal, decision, reaction_time, rng_states): 
    """ Cuda kernel for model with drift, leak, lateral inhibition (competiting), ddm like stimulus driven inhibition and urgency model. 
    Also works with within-trial dynamic stimulus, boundary and drift variability. Additionally, implementation of starting point offset and drift rate offset for bias terms.
    
    Note: Since cuda kernels cannot return values, we pass output variables to in function as mutable variables.
    
    Args:
        stimulus: Stimulus intensities with shape (num_trials, num_choices, num, time). Type: array(3D, float32)
        starting_point: 1D array of starting point for each accumulator with length equal to number of time steps. Type: array(1D, float32) 
        drift_gain: Common drift rate of accumulators (same for all accumulators). Type: float32
        drift_variability: Time-dependent drift-variability variability for each time point with length equal to number of time steps. Type array(1D, float32)
        drift_offset: Offset in drift-rate i.e., drift-rate in case of no/random stimulus with length equal to number of choices. Type: array(1D, float32) 
        decision_bound: Time dependent decision-bound. Array of length equal to number of time steps. Type: array(1D, float32)
        nondecision_time: Non-decision time. Type: float32
        lateral_inhibition: Waight of lateral inhibition from other accumulator. Type: float32
        leak: Leak factor from indivdual accumulator. Proportional to evidence in respective accumulator. Type: float32
        neural_ddm: DDM like lateral inhibition from drift-rate from other stimulus intensities. Type: bool
        urgency_signal: Boolean for linerly time-dependent urgency signal. Type: bool
        rng_states: random normal generator for cuda. Must have same number of states as parellelization (i.e., num_trials)
        
    Return:
        decision: Decision on each trial as a number of accumulator (0:num_choices-1) with length equal to number of trials. Type: array(1D, float32)
        reaction_time: Reaction time on each trial with length equal to number of trials. Type: array(1D, float32)       
    

    Information:
        2022-03 VT wrote it
        
    """

    
    # Initializing local variables
    decision_variable = cuda.local.array(num_choices, dtype=numba.float32)
    drift_rate = cuda.local.array(num_choices, dtype=numba.float32)

    # Input grid and stride size
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    # Loop over number of trials (parellel)
    for tr in range(start, stimulus.shape[0], stride): 
        
        # Instantiating decision variables with starting point
        for accumulator in range(stimulus.shape[1]):
            decision_variable[accumulator] = starting_point[accumulator]        
        sum_decision_variables = summation_cuda(decision_variable)
        boundary_reached = False
        
        # Loop over number of samples
        for t in range(stimulus.shape[2]):
            vector_scaling_cuda(stimulus[tr,:,t], drift_gain, drift_rate)             
            sum_decision_variables = summation_cuda(decision_variable)
            
            # Loop over number of choices (or accumulator)
            for accumulator in range(stimulus.shape[1]):                
                diffusion_step = drift_rate[accumulator] + drift_offset[accumulator] + (cuda_random.xoroshiro128p_normal_float32(rng_states, tr)*drift_variability[t])                  
                leak_step = leak * decision_variable[accumulator]
                lateral_dv_inhibition_step = lateral_inhibition * (sum_decision_variables - decision_variable[accumulator])  # Lateral inhibition from all decision_variables except self    
                ddm_like_dr_inhibition_step = (summation_cuda(drift_rate) + summation_cuda(drift_offset)) - (drift_rate[accumulator] - drift_offset[accumulator])  # Collecting all drift-rate except current
                
                decision_step = decision_variable[accumulator] \
                                + diffusion_step \
                                - leak_step \
                                - lateral_dv_inhibition_step \
                                - ddm_like_dr_inhibition_step*neural_ddm
                
                if urgency_signal == True:
                    decision_step*=t
                         
                decision_variable[accumulator] = ReLU_cuda(decision_step)
                
                if decision_variable[accumulator] > decision_bound[t]:
                    boundary_reached = True
                    decision[tr] = accumulator
                    reaction_time[tr] = t + nondecision_time
                    break
                    
            if boundary_reached == True:
                break    
        


def IAM_gpu_sim(stimulus, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time, lateral_inhibition, leak, neural_ddm, urgency_signal, batch_size=None, seed=None):
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
        lateral_inhibition: Waight of lateral inhibition from other accumulator. Type: float32
        leak: Leak factor from indivdual accumulator. Proportional to evidence in respective accumulator. Type: float32
        neural_ddm: DDM like lateral inhibition from drift-rate from other stimulus intensities. Type: bool
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
           
    if starting_point.shape[0] != stimulus.shape[1]:        
        raise ValueError('Must provide starting point as array with length eqaul to num_choices')
        
    if drift_variability.shape[0] != stimulus.shape[2]:        
        raise ValueError('Must provide drift variability as array with length eqaul to num_samples')
        
    if drift_offset.shape[0] != stimulus.shape[1]:        
        raise ValueError('Must provide drift offset as array with length eqaul to num_choices')
        
    if decision_bound.shape[0] != stimulus.shape[2]:        
        raise ValueError('Must provide decision bound as array with length eqaul to num_samples')
        
        
    
    # If batch_size not provided, simulate whole data
    if batch_size is None:
        batch_size = stimulus.shape[0]

    # Setting up parellel grid
    multiplier = 8
    blockdim = int(multiplier*32)
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
        IAM_kernel[griddim, blockdim](stimulus_batch, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time, lateral_inhibition, leak, neural_ddm, urgency_signal, decision_cp, reaction_time_cp, rng_states) 
        cuda.synchronize()
        
        decision_np = np.append(decision_np, cp.asnumpy(decision_cp))
        reaction_time_np = np.append(reaction_time_np, cp.asnumpy(reaction_time_cp))
        
    return decision_np, reaction_time_np




def initialize_IAM_kernel(num_trials=3, num_choices=2, num_samples=1000):
    
    """
    Initializing GPU kernels for Indiavidual accumulator model if GPU is available
    
    Args:
        is_gpu_available: Boolean
        
    Return:
        None
    """

    coherence = np.ones((num_trials,num_samples))*50     # 100

    starting_point = np.zeros(num_choices, dtype=float32)             
    drift_offset = np.zeros(num_choices, dtype=float32)
    drift_gain = float32(10e-5)             # drift gain
    drift_variability = float32(10e-3)      # diffusion variability
    nondecision_time = float32(100)         # Non-decision time (msec)
    decision_bound = 1
    bound_rate = 0
    bound_delay = 0
    lateral_inhibition = 0.005
    leak = 0.001
    neural_ddm = 0.2
    urgency_signal = False
    # Dynamic time-dependent variables
    stimulus_cp = cp.asarray(get_unsigned_coherence_matrix(coherence))
    decision_bound_cp = get_time_dependent_bound(decision_bound, bound_rate, bound_delay)
    drift_variability_cp = get_time_dependent_variability(drift_variability)

    decision = cp.empty(stimulus_cp.shape[0])*cp.NaN
    reaction_time = cp.empty(stimulus_cp.shape[0])*cp.NaN

    blockdim = (256)
    griddim = stimulus_cp.shape[0] // (blockdim) + 1
    rng_states = cuda_random.create_xoroshiro128p_states(int(np.prod(blockdim) * np.prod(griddim)), seed=3)  # useful in case of 2D grid. Normal version would be "blockdim * griddim"
    cuda.synchronize()
    individual_accumulator_kernel[griddim, blockdim](stimulus_cp, starting_point, drift_gain, drift_variability_cp, drift_offset, decision_bound_cp, nondecision_time, lateral_inhibition, leak, neural_ddm, urgency_signal, decision, reaction_time, rng_states) 
    cuda.synchronize()



                                    