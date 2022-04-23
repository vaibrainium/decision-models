import numpy as np
import unittest
from src.DDM import *
from src.IAM import *
from src.dynamic_parameters import *
from src.likelihood_estimation import *




class TestModel(unittest.TestCase):
	def test_QLME(self):

		"""
		Makes sure likelihoods (-llh) are calculated correctly by QMLE method.
		"""

		RT_data = np.arange(1,3,1/100)
		RT_model = np.arange(2.1,4,1/100)
		nLLh1 = get_llh_QMLE(RT_model, RT_data, nbins=20)

		RT_data = np.arange(2,4,1/100)
		RT_model = np.arange(2.1,4,1/100)
		nLLh2 = get_llh_QMLE(RT_model, RT_data, nbins=20)

		self.assertGreater(nLLh1, 0)
		self.assertGreater(nLLh2, 0)
		self.assertGreater(nLLh1, nLLh2)


	def test_PDE(self):

		"""
		Makes sure likelihoods (-llh) are calculated correctly by PDE method.
		"""

		RT_data = np.arange(1,3,1/100)
		RT_model = np.arange(2.1,4,1/100)
		nLLh1 = get_llh_PDE(RT_model, RT_data)

		RT_data = np.arange(2,4,1/100)
		RT_model = np.arange(2.1,4,1/100)
		nLLh2 = get_llh_PDE(RT_model, RT_data)

		self.assertGreater(nLLh1, 0)
		self.assertGreater(nLLh2, 0)
		self.assertGreater(nLLh1, nLLh2)



if __name__ == '__main__':
    unittest.main()