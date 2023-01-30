"""
# delta_Wq_statistics.py is a part of the EMD4CPV package.
# Copyright (C) 2023 EMD4CPV authors (see AUTHORS for details).
# EMD4CPV is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

from delta_Wq import *
from energy_test import compute_T

class delta_Wq_statistics(delta_Wq):
	"""
	This class provides functionality for computing test statistics using the generated dWq data from delta_Wq.py. 
	"""
	def compute_Wq(self, dWq, q):
		"""
		Compute Wq by summing given delta Wq values and taking the inverse root of q

		dWq (numpy array): delta Wq values

		q (float):  Wasserstein parameter

		Returns Wq as a float
		"""
		return np.sum(dWq)**(1/q)

	def compute_log_Wq(self, dWq, q):
		"""
		Compute Wq by summing given delta Wq values and taking the inverse root of q

		dWq (numpy array): delta Wq values

		q (float):  Wasserstein parameter

		Returns Wq as a float
		"""
		log_dWq = np.log(dWq)

		return np.sum(log_dWq)

	def compute_YWq(self, dWq, minimum, maximum):
		"""
		Compute the windowed-Wq by only counting values with the defined minimum and maximum 

		dWq (numpy array): delta Wq values

		minimum (float): window minimum

		maximum (float): window minimum

		Returns sum of window contributions as a float
		"""
		YWq = []
		for i in range(len(dWq)):
			if dWq[i] >= minimum and dWq[i] <= maximum:
				YWq.append(1)
		return np.sum(YWq)

	def compute_MWq(self, dWq, min_window, max_window, min_sill, max_sill):
		"""
		Wq statistic allows for a single window ( +1 positive contribution) and a single sill ( -1 negative contribution)

		dWq (numpy array): delta Wq values

		min_window (float): window minimum

		max_window (float): window minimum

		min_sill (float):   sill minimum

		max_sill (float):   sill maximum

		Returns sum of window-sill contributions as a float
		"""
		MWq = []
		for i in range(len(dWq)):
			MWq.append(self.window(dWq[i], edge_min = min_window, edge_max = max_window) + self.sill(dWq[i], edge_min = min_sill, edge_max = max_sill))
		return np.sum(MWq)

	def compute_IWq(self, dWq, q, window_sill_dictionary):
		"""
		Improved-window-sill Wq statistic allowing for any number of windows or sills which contribute +1 if in the windowed regions or -1 if in the sill regions.

		dWq (numpy array): delta Wq values

		window_sill_dictionary: this dictionary contains all of the window-sill data the convention is as follows 
								windnow_sill_dictionary = {'window':[[window_min_1, window_max_1], [window_min_2, window_max_2], ...], 'sill': [[sill_min_1, sill_max_1], [sill_min_2, sill_max_2], ...]}

		Returns sum of all window-sill contributions
		"""
		try:
			n_window = len(window_sill_dictionary['window'])
		except:
			n_window = 0
		try:
			n_sill = len(window_sill_dictionary['sill'])
		except:
			n_sill = 0
		
		IWq = []
		for i in range(len(dWq)):
			for j in range(n_window):
				# Count +1 if point lies in window region
				if dWq[i] >= window_sill_dictionary['window'][j][0] and dWq[i] <= window_sill_dictionary['window'][j][1]:
					IWq.append(1)
			for k in range(n_sill):
				# Count -1 if point lies in sill region
				if dWq[i] >= window_sill_dictionary['sill'][k][0] and dWq[i] <= window_sill_dictionary['sill'][k][1]:
					IWq.append(-1)
		return np.sum(IWq)

	def compute_log_IWq(self, dWq, q, window_sill_dictionary):
		"""
		Improved-window-sill Wq statistic allowing for any number of windows or sills which contribute +1 if in the windowed regions or -1 if in the sill regions.

		dWq (numpy array): delta Wq values

		window_sill_dictionary: this dictionary contains all of the window-sill data the convention is as follows 
								windnow_sill_dictionary = {'window':[[window_min_1, window_max_1], [window_min_2, window_max_2], ...], 'sill': [[sill_min_1, sill_max_1], [sill_min_2, sill_max_2], ...]}

		Returns sum of all window-sill contributions
		"""
		try:
			n_window = len(window_sill_dictionary['window'])
		except:
			n_window = 0
		try:
			n_sill = len(window_sill_dictionary['sill'])
		except:
			n_sill = 0
		
		log_dWq = np.log(dWq.astype(float))

		IWq = []
		for i in range(len(log_dWq)):
			for j in range(n_window):
				# Count +1 if point lies in window region
				if log_dWq[i] >= window_sill_dictionary['window'][j][0] and log_dWq[i] <= window_sill_dictionary['window'][j][1]:
					IWq.append(1)
			for k in range(n_sill):
				# Count -1 if point lies in sill region
				if log_dWq[i] >= window_sill_dictionary['sill'][k][0] and log_dWq[i] <= window_sill_dictionary['sill'][k][1]:
					IWq.append(-1)
		return np.sum(IWq)

	def compute_exp_log_IWq(self, dWq, q, window_sill_dictionary):
		"""
		Improved-window-sill Wq statistic allowing for any number of windows or sills which contribute +1 if in the windowed regions or -1 if in the sill regions.

		dWq (numpy array): delta Wq values

		window_sill_dictionary: this dictionary contains all of the window-sill data the convention is as follows 
								windnow_sill_dictionary = {'window':[[window_min_1, window_max_1], [window_min_2, window_max_2], ...], 'sill': [[sill_min_1, sill_max_1], [sill_min_2, sill_max_2], ...]}

		Returns sum of all window-sill contributions
		"""
		# Collect information on the number of windows and anti-windows
		try:
			n_window = len(window_sill_dictionary['window'])
		except:
			n_window = 0
		try:
			n_sill = len(window_sill_dictionary['sill'])
		except:
			n_sill = 0
		
		# Get the log of the values
		log_dWq = np.log(dWq)

		IWq = []
		for i in range(len(log_dWq)):
			for j in range(n_window):
				# Count +1 if point lies in window region
				if log_dWq[i] >= window_sill_dictionary['window'][j][0] and log_dWq[i] <= window_sill_dictionary['window'][j][1]:
					IWq.append(log_dWq[i])
			for k in range(n_sill):
				# Count -1 if point lies in sill region
				if log_dWq[i] >= window_sill_dictionary['sill'][k][0] and log_dWq[i] <= window_sill_dictionary['sill'][k][1]:
					IWq.append(-log_dWq[i])
		
		# Exponentiate back to the orginal Wq values and sum
		IWq = np.array(IWq)
		IWq = np.exp(IWq)

		return np.sum(IWq)

	def statistic_dict(self):
		"""
		Create dictionary which points to the relevant statistic computation given the statistic key
		"""
		return {'Wq': self.compute_Wq, 'Wq_binned': self.compute_Wq, 'YWq': self.compute_YWq, 'MWq': self.compute_MWq, 'IWq': self.compute_IWq, 'log_Wq': self.compute_log_Wq, 'log_IWq': self.compute_log_IWq, 'exp_log_IWq':self.compute_exp_log_IWq, 'T': compute_T}

	def get_statistic(self, dWq, statistic_key, statistic_parameters):
		'''
		Given a statistic key this function actually collects the data from the executed statistic function

		dWq (numpy array): delta Wq values

		statistic_key (string): statistic to be computed, must match key in statistic dict

		statistic_parameters (list, dict): in order for more flexibility and ultimately ease of use, any statistic parameters must be input as a list 
										   (even if there is only one component)

		Returns specified statistic as a float
		'''
		return self.statistic_dict()[statistic_key](dWq, *statistic_parameters)

	def get_statistic_list(self, dWq_list, statistic_key, statistic_parameters):
		'''
		Given a statistic key and many lists of dWq data this function compute specified statistic for each list of dWq

		dWq_list (numpy array of numpy arrays): list of list of delta Wq values

		statistic_key (string): statistic to be computed, must match key in statistic dict

		statistic_parameters (list, dict): in order for more flexibility and ultimately ease of use, any statistic parameters must be input as a list 
										   (even if there is only one component)

		Returns numpy array of floats of the specified statistic
		'''
		statistic_list = []
		# Compute statistic for all dWq
		for i in range(len(dWq_list)):
				statistic_list.append(self.get_statistic(dWq_list[i], statistic_key, statistic_parameters))
		return np.array(statistic_list)

	def load_dWq(self, dWq_path):
		"""
		Load in the relevant delta Wq data file to be used in the computation of statistics

		dWq_path (string): path to delta Wq data file
		"""
		if os.path.exists(dWq_path) == False:
			print(rf'delta_W data for q = {self.q} does not exist. Run generate_dWq first.')
		else:
			dWq = []
			with open(dWq_path, 'r') as csvfile:
				# Create csv reader object
				csvreader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
				# Extract dWq data row by row
				for row in csvreader:
					dWq.append(row)
		return np.array(dWq, dtype=object)

	def write_statistic_data(self, dWq_path, statistic_manual, write_path = None):
		"""
		Given the path to a dWq data file containing many lists of delta Wq data, compute the relevant statistic given in the 
		statistic manual and return/write them

		dWq_path (string or list): path to delta_Wq data (strings ordered by q values as defined in class instance)

		statistic_manual (dictionary): dictionary containing statistics to be computed + all neccesary parameters. 
									   The dict format convention is given by, for example,
									   statistic_manual = {'statistic': [[statistic_parameters_q[0]], [statistic_parameters_q[1]], [...] ], 'statistic_2': [...], ...}
		"""
		keys = statistic_manual.keys()
		# For ease of use the user may input a list 
		if type(self.q) == list:
			for key in keys:
				for i in range(len(self.q)):
					if write_path == None and i == 0: write_path = ['./' + key + rf'_{self.q[j]}_data.txt' for j in range(len(self.q))] 
					dWq = self.load_dWq(dWq_path[i])
					# Compute the statistic
					statistic = self.get_statistic_list(dWq, key, statistic_manual[key][i])
					# Open the file in the write mode
					print('Writing...')
					with open(write_path[i], 'w') as f:
					    # Create the csv writer
					    writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
					    # Write statistic data
					    for i in range(len(statistic)):
					    	writer.writerow([statistic[i]])
		else:
			for key in keys:
				if write_path == None: write_path = './' + key + rf'_{self.q}_data.txt'
				dWq = self.load_dWq(dWq_path)
				statistic = self.get_statistic_list(dWq, key, statistic_manual[key])
				# Open the file in the write mode
				with open(write_path, 'w') as f:
				    # Create the csv writer
				    writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
				    # Write statistic data
				    for i in range(len(statistic)):
				    	writer.writerow([statistic[i]])
