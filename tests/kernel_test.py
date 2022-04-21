import numpy as np
import unittest
from src.DDM import *
from src.IAM import *
from src.dynamic_parameters import *




class TestModel(unittest.TestCase):
	def test_DDM(self):

		"""
		Makes sure that cuda execution of DDM model matches the CPU simulations
		"""

		num_choices = 2
		num_trials = 3
		num_samples = 2000

		coherence = np.ones((num_trials,num_samples))*50     # 100
		coherence[0,0:300] = -50
		coherence[0,1000:1300] = 150
		coherence[1,10:350] = -50
		coherence[1,800:1350] = -150
		coherence[2,10:350] = -100
		coherence[2,350:700] = 120

		starting_point = 0 #np.array(np.zeros(1), dtype=float32)             
		drift_offset = 0 #np.array(np.zeros(1), dtype=float32)
		drift_gain = np.float32(5e-5)             # drift gain
		drift_variability = np.float32(0)#10e-3)      # diffusion variability
		nondecision_time = np.float32(100)         # Non-decision time (msec)
		decision_bound = 1
		bound_rate = 0
		bound_delay = 0
		lateral_inhibition = 0
		leak = 0
		neural_ddm = 0
		urgency_signal = False
		# Dynamic time-dependent variables
		stimulus = get_unsigned_coherence_matrix(coherence)
		stimulus_cp= cp.asarray(stimulus)
		decision_bound_cp = get_time_dependent_bound(decision_bound, bound_rate, bound_delay, stop_time=num_samples)
		decision_bound = cp.asnumpy(decision_bound_cp)
		drift_variability_cp = get_time_dependent_variability(drift_variability, time_coefficient=0, stop_time=num_samples)
		drift_variability = cp.asnumpy(drift_variability_cp)

		decision_cpu, reaction_time_cpu = DDM_cpu_sim(stimulus, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time)
		decision_gpu, reaction_time_gpu = DDM_gpu_sim(stimulus_cp, starting_point, drift_gain, drift_variability_cp, drift_offset, decision_bound_cp, nondecision_time, urgency_signal)

		self.assertSequenceEqual(decision_cpu.tolist(), decision_gpu.tolist())
		self.assertSequenceEqual(reaction_time_cpu.tolist(), reaction_time_gpu.tolist())





	def test_IAM(self):

		"""
		Makes sure that cuda execution of Independent Accumulator Model matches the CPU simulations
		"""

		num_choices = 2
		num_trials = 3
		num_samples = 2000

		coherence = np.ones((num_trials,num_samples))*50     # 100
		coherence[0,0:300] = -50
		coherence[0,1000:1300] = 150
		coherence[1,10:350] = -50
		coherence[1,800:1350] = -150
		coherence[2,10:350] = -100
		coherence[2,350:700] = 120

		starting_point = np.zeros(num_choices, dtype=np.float32)             
		drift_offset = np.zeros(num_choices, dtype=np.float32)
		drift_gain = np.float32(10e-5)             # drift gain
		drift_variability = np.float32(0)      # diffusion variability
		nondecision_time = np.float32(100)         # Non-decision time (msec)
		decision_bound = 1
		bound_rate = 0
		bound_delay = 0
		lateral_inhibition = 0.005
		leak = 0.01
		neural_ddm = 0.2
		urgency_signal = False
		# Dynamic time-dependent variables
		stimulus = get_unsigned_coherence_matrix(coherence)
		stimulus_cp= cp.asarray(stimulus)
		decision_bound_cp = get_time_dependent_bound(decision_bound, bound_rate, bound_delay, stop_time=num_samples)
		decision_bound = cp.asnumpy(decision_bound_cp)
		drift_variability_cp = get_time_dependent_variability(drift_variability, time_coefficient=0, stop_time=num_samples)
		drift_variability = cp.asnumpy(drift_variability_cp)
		                
		decision_cpu, reaction_time_cpu = IAM_cpu_sim(stimulus, starting_point, drift_gain, drift_variability, drift_offset, decision_bound, nondecision_time, lateral_inhibition, leak, neural_ddm, urgency_signal)
		decision_gpu, reaction_time_gpu = IAM_gpu_sim(stimulus_cp, starting_point, drift_gain, drift_variability_cp, drift_offset, decision_bound_cp, nondecision_time, lateral_inhibition, leak, neural_ddm, urgency_signal)

		self.assertSequenceEqual(decision_cpu.tolist(), decision_gpu.tolist())
		self.assertSequenceEqual(reaction_time_cpu.tolist(), reaction_time_gpu.tolist())


if __name__ == '__main__':
    unittest.main()