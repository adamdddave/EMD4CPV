"""
# delta_Wq.py is a part of the EMD4CPV package.
# Copyright (C) 2023 EMD4CPV authors (see AUTHORS for details).
# EMD4CPV is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

import pandas as pd
import numpy as np
np.set_printoptions(precision=16)
import wasserstein
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from random import shuffle
import seaborn as sns
import sys
import random
import csv
import os
#from rich.progress import track
#from tqdm.rich import tqdm
from tqdm import tqdm
from scipy import stats

sns.set()
sns.set_style("ticks")
sns.set_context("paper", font_scale = 3.0)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 25


class delta_Wq:	
	def __init__(self, X_data, Xbar_data, N_len, Nbar_len, q, mass, pool_len = None, shuffle_or_master = 'master'):
		"""
		This class contains routines used for computing and outputting data used in the Dalitz plot analyses for CP violation.
		
		X_data (string): -- Path to file cotaining all Dalitz plot points for first decay dataset. Our convention assumes the data has 
							been prepared such that each line contains three entries corresponding to m23_i^2, m13_i^2, m12_i^2 data 
							with a single space between each entry.

		Xbar_data (string): Path to file containing all Daltiz plot points for second (conjugate) decay dataset. This file is also 
							assumed to follow the same conventions as mention above.

		N_len (int): ------ Desired length for first dataset

		Nbar_len (int): --- Desired length for second dataset

		q (float): -------- Wasserstein distance parameter

		pool_len (int):	--- Total size in bytes of X and Xbar datasets. If None the program will compute the length automatically
							 (Generalization to two data pools with different lengths is straight-forward, TBD)

		mass (float): ----- Mass of decaying particle to be used in normalizing the cost (distance) matrix

		shuffle_or_master (string) : Desired method for computing the CP-conserving distribution. Shuffle uses the same two datasets
									 and permutes these two datasets within each other whereas master samples new datasets randomly
									 each iteration from the larger data pool.
		"""
		self.X_data    = X_data
		self.Xbar_data = Xbar_data
		self.N_len     = N_len
		self.Nbar_len  = Nbar_len
		self.q         = q
		self.mass      = mass
		self.shuffle_or_master = shuffle_or_master
		if pool_len == None: 
			# Adjustment factor as to not reach the end of the file
			self.pool_len = os.path.getsize(self.X_data) - 500
		else: self.pool_len = pool_len

	def get_subset(self, data, length):
		"""
		Generate a random list of data points with length len from the data pool. This algorithm ensures that
		no duplicates are chosen, which becomes important when obtaining the CP conserving probability 
		distributions via the permutation method. This algorithm will need to be modified for cases in which 
		the len of the desired samples is ~ the size of the data pool. In these cases the algorithm reaches
		a point where it is often sampling repeat values and thus performs many unsuccessful trials. 

		data (string): path to dataset following the conventions described in the intialization of the class

		len (int): --- length of desired list of data points

		returns numpy array of m2 data
		"""
		m2 = []
		#location = random.sample(range(self.pool_len), length)
		with open(data, 'r') as f:
			for i in range(length):
				if i == 0:
					location = random.randrange(self.pool_len)
					# Go to location in file and place marker
					f.seek(location)
					# Bound to partial line
					f.readline()
					# Split masses into list
					ls = f.readline().split()
					m2.append([float(ls[0]), float(ls[1]), float(ls[2])])
				else:
					# Generate a trial mass value
					location = random.randrange(self.pool_len)
					# Go to location in file and place marker
					f.seek(location)
					# Bound to partial line
					f.readline()
					# Split masses into list
					ls = f.readline().split()
					try:
						m2_i = [[float(ls[0]), float(ls[1]), float(ls[2])]]
					except:
						print('End of file reached at location ', location, '. Trying reducing the pool_len, retrying...')
						# Generate a trial mass value
						location = random.randrange(self.pool_len)
						# Go to location in file and place marker
						f.seek(location)
						# Bound to partial line
						f.readline()
						# Split masses into list
						ls = f.readline().split()
						m2_i = [[float(ls[0]), float(ls[1]), float(ls[2])]]
					
					if not(any(i in m2_i for i in m2)):
						m2.append(m2_i[0])
					else:
						# You found a duplicate, loop until you find an entry which is unique
						while any(i in m2_i for i in m2):
							# Generate a trial mass value
							location = random.randrange(self.pool_len)
							# Go to location in file and place marker
							f.seek(location)
							# Bound to partial line
							f.readline()
							# Split masses into list
							ls = f.readline().split()
							try:
								m2_i = [[float(ls[0]), float(ls[1]), float(ls[2])]]
							except:
								continue
						m2.append(m2_i[0])
			
			"""
			# This way works but is x10 slower than the above method
			m2 = random.sample(f.readlines(), length)
			# Transform into list of floats
			for i in range(length):
				ls = m2[i].split()
				m2[i] = np.array([float(ls[0]), float(ls[1]), float(ls[2])])
			"""
		return np.array(m2)

	def get_cost(self, q, X_data, Xbar_data):
		return ((1/self.mass**2) * cdist(X_data, Xbar_data, 'euclidean'))**q

	def get_Wq(self, q , k = 1, kbar = None, X_data = None, Xbar_data = None):
		if X_data == None and Xbar_data == None:
			X_data = self.get_subset(self.X_data, self.N_len)
			Xbar_data = self.get_subset(self.Xbar_data, self.Nbar_len)
		if kbar == None: kbar = k
		# Compute the cost matrix
		cost = self.get_cost(q, X_data, Xbar_data)
		# Define the weights
		weights_X = k * np.ones(len(X_data))*1./float(len(X_data))
		weights_Xbar = kbar * np.ones(len(Xbar_data))*1./float(len(Xbar_data))
		# Compute the EMD
		emd = wasserstein.EMD(n_iter_max=int(1e9))
		emd_val = emd(weights_X, weights_Xbar, cost)
		return emd_val**(1/q)

	def compute_dWq(self, q, X_m2_data, Xbar_m2_data, weights_X, weights_Xbar, N = None, Nbar = None, with_masses = False):
		"""
		Compute delta Wq data

		q (float, or list of floats): Wasserstein distance parameter

		X_m2_data (numpy array): ---- Array containing Dalitz points 

		Xbar_m2_data (numpy array): - Array containing conjugate Daltiz points

		weights_X (numpy array): ---- Array containing X weights (or 'mass')

		weights_Xbar (numpy array): - Array containing Xbar weights (or 'mass')

		N (int): -------------------- Length of first dataset. If None, set to initialized class value self.N_len

		Nbar (int):	----------------- Length of second dataset. If None, set to initialized class value self.Nbar_len

		with_masses (bool): --------- If true return delta Wq data as well as mass data (typically used for creating
									  visualizations of the Wq asymmetry)		

		"""
		# If nothing is given set to defualt class variable
		if N == None: N = self.N_len 
		if Nbar == None: Nbar = self.Nbar_len

		# Compute the cost matrix
		try:
			cost = ((1/self.mass**2) * cdist(X_m2_data, Xbar_m2_data, 'euclidean'))**q
		except:
			# Catch the case where 1D input is provided
			cost = cdist(np.column_stack((X_m2_data, np.zeros(len(X_m2_data)))), np.column_stack((Xbar_m2_data, np.zeros(len(Xbar_m2_data)))), 'euclidean')

		# Compute the Wasserstein distance following the notation defined in the paper
		emd = wasserstein.EMD(n_iter_max=int(1e9), norm = True)
		emd_val = emd(weights_X, weights_Xbar, cost)
	
		# Get the flow information telling us which elements are paired in the computation
		flows = emd.flows()
	
		# Compute delta W
		if with_masses == False:
			dWq = []
			for i in range(N):
				for j in range(Nbar):
					f_ij = flows[i,j]
					
					#if f_ij > 0.0 and cost[i,j]==0.0: print(i,j, X_m2_data[i], Xbar_m2_data[j])
					if f_ij > 0.0 and cost[i,j] > 0.0:
						dWq.append(f_ij * cost[i,j])
			return np.array(dWq)
		else:
			dWq, m2_B, m2_Bbar, pos_B, pos_Bbar = [],[],[],[],[]
			for i in range(N):
				for j in range(Nbar):
					f_ij = flows[i,j]
					if f_ij > 0.0:
						dWq.append(f_ij * cost[i,j])
						m2_B.append(np.array(X_m2_data[i]))
						m2_Bbar.append(np.array(Xbar_m2_data[j]))
						pos_B.append(i)
						pos_Bbar.append(j)

			return np.array(dWq), np.array(m2_B), np.array(m2_Bbar), np.array(pos_B), np.array(pos_Bbar)

	def swap(self, n_swap, A, B, rand_A, rand_B):
		"""
		Swap the elements of A and B according to rand_A and rand_B

		A (numpy array): Array to be swapped
		B (numpy array): Array to be swapped
		rand_A (numpy array): Randomly initialized array containing 
		"""
		# Swap the elements
		for i in range(n_swap):
			B_copy = B[rand_B[i]].copy()
			B[rand_B[i]] = A[rand_A[i]]
			A[rand_A[i]] = B_copy
		return A, B

	def split(self, data, size):
		"""
		Split the given list of data into a list of lists where the
		nested lists are of length size.
		Ex.
		split([1, 2, 3, 4], 2) -> [[1, 2], [3, 4]]
		"""
		elements = []
		while len(data) > size:
			pice = data[:size]
			elements.append(pice)
			data = data[size:]
		elements.append(data)
		return np.array(elements)

	def generate_dWq(self, n_perm, q, dWq_file_name = None, save = False):
		"""
		Generate the dWq and write to dWq_file_name 
		"""
		dWq = []
		# Define the weights
		weights_X = np.ones(self.N_len)*1./float(self.N_len)
		weights_Xbar = np.ones(self.Nbar_len)*1./float(self.Nbar_len)

		# We have two different methods of generating the CPC PDFs, shuffling CPV + CPC or by sampleing directly from two CPC datasets
		if self.shuffle_or_master == 'master':
			# Generate the permutation data
			for i in tqdm(range(n_perm), ncols = 100, ascii = "░▒█"):
			#for i in track(range(n_perm), disable = True):
				# Grab a subset of the larger data file
				X_m2_perm    = self.get_subset(self.X_data, self.N_len)
				Xbar_m2_perm = self.get_subset(self.Xbar_data, self.Nbar_len)
				
				# Compute delta W 
				dWq_i = self.compute_dWq(q, X_m2_perm, Xbar_m2_perm, weights_X, weights_Xbar)
	
				# Write out the dW data
				if save == True:
					# Open a file and record the EMD_i data
					with open(dWq_file_name, 'a') as f:
						# Create the csv writer
						writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
						# Write dW's
						writer.writerow(dWq_i)
				dWq.append(np.array(dWq_i))
			return np.array(dWq)
		else:
			# Grab a subset of the larger data file
			X_m2    = self.get_subset(self.X_data, self.N_len)
			Xbar_m2 = self.get_subset(self.Xbar_data, self.Nbar_len)
			assert(not(any(i in X_m2 for i in Xbar_m2)))

			# Generate the permutation data from just the above to files
			for i in tqdm(range(n_perm), ncols = 100, ascii = "░▒█"):
				# Work with copies to avoid in place memory issues
				X_m2_copy = X_m2.copy()
				Xbar_m2_copy = Xbar_m2.copy()
				
				# Shuffle the datasets
				X_Xbar_m2_shuffle = np.vstack((X_m2_copy, Xbar_m2_copy))
				X_Xbar_m2_shuffle = np.random.permutation(X_Xbar_m2_shuffle)

				# Split the shuffled data
				X_Xbar_m2_split = np.array_split(X_Xbar_m2_shuffle, [self.N_len], axis = 0)
				
				X_m2_perm = np.array(X_Xbar_m2_split[0])
				Xbar_m2_perm = np.array(X_Xbar_m2_split[1])
				
				# Compute delta W 
				dWq_i = self.compute_dWq(q, X_m2_perm, Xbar_m2_perm, weights_X, weights_Xbar)
				
				# Write out the dW data
				if save == True:
					# Open a file and record the EMD_i data
					with open(dWq_file_name, 'a') as f:
						# Create the csv writer
						writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
						# Write dW's
						writer.writerow(dWq_i)
				dWq.append(np.array(dWq_i))
			return np.array(dWq)

	def get_dWq(self, n_perm, dWq_file_name = None, save = False):
		"""
		Generate dWq values for all input q values
		"""
		if type(self.q) == list:
			dWq = []
			for i in range(len(self.q)):
				print('q = ', self.q[i])
				dWq.append(self.generate_dWq(n_perm, self.q[i], dWq_file_name[i], save))
			return np.array(dWq)
		else:
			return self.generate_dWq(n_perm, self.q, dWq_file_name, save)


	#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
	#------------------------------------------------ Binned Wasserstein Distance -------------------------------------------------#
	#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#

	def get_binned_subset(self, X_m2_data, Xbar_m2_data, n_bins, min_max_dictionary = None):
		"""
		Generate binned subset data from provided Daltiz datasets. Note the the returned mass^2 Dalitz variables
		for X and Xbar will be equivalent (set to the centers of the computed bins). The weights, however, 
		are distinct. 

		X_m2_data (numpy array): ------- Array containing Dalitz points to be binned

		Xbar_m2_data (numpy array): ---- Array containing conjugate Daltiz points to be binned

		n_bins (int): ------------------ Desired number of bins

		min_max_dictionary (dictionary): Dictionary containing global minimum and maximum data from the
										 larger data pool. This is done to avoid problems with 
										 bin edges. If None is passed the global min and max
										 will be computed however, for sufficiently large datasets, this
										 should be set manually to avoid large run times. 

		Return the new binned mass values, weights, and new dataset lengths
		"""

		# Perform a 3-dimensional binning - easily generalizable to n-dimensional binning
		H_X, edges_X, binnumber_X = stats.binned_statistic_dd([X_m2_data[:,0], X_m2_data[:,1], X_m2_data[:,2]], values = None, statistic = 'count', bins=n_bins, range = [min_max_dictionary['m23'], min_max_dictionary['m13'], min_max_dictionary['m12']], expand_binnumbers=True)
		H_Xbar, edges_DO_NOT_USE, binnumber_Xbar = stats.binned_statistic_dd([Xbar_m2_data[:,0], Xbar_m2_data[:,1], Xbar_m2_data[:,2]], values = None, statistic = 'count', bins=[edges_X[0], edges_X[1], edges_X[2]], expand_binnumbers=True)
		
		assert len(edges_X[0]) == len(edges_X[1]) == len(edges_X[2])

		# Find the new bin centers
		bin_centers_X_m23      = [edges_X[0][i] for i in range(len(edges_X[0])-1)]
		bin_centers_X_m13      = [edges_X[1][i] for i in range(len(edges_X[0])-1)]
		bin_centers_X_m12      = [edges_X[2][i] for i in range(len(edges_X[0])-1)]
		bin_centers_Xbar_m23   = [edges_X[0][i] for i in range(len(edges_X[0])-1)]
		bin_centers_Xbar_m13   = [edges_X[1][i] for i in range(len(edges_X[0])-1)]
		bin_centers_Xbar_m12   = [edges_X[2][i] for i in range(len(edges_X[0])-1)]
		
		# Reassign new m2 values equal to the centers of the bins, only record non-zero bins
		X_m2_binned = [np.array([bin_centers_X_m23[i], bin_centers_X_m13[j], bin_centers_X_m12[k]]) for i in range(n_bins) for j in range(n_bins) for k in range(n_bins) if H_X[i][j][k] > 0 and H_Xbar[i][j][k] > 0]
		Xbar_m2_binned = [np.array([bin_centers_Xbar_m23[i], bin_centers_Xbar_m13[j], bin_centers_Xbar_m12[k]]) for i in range(n_bins) for j in range(n_bins) for k in range(n_bins) if H_X[i][j][k] > 0 and H_Xbar[i][j][k] > 0]

		# The weights are now given by some local count density dependent on the size of the bins
		weights_X = np.array([H_X[i][j][k] for i in range(n_bins) for j in range(n_bins) for k in range(n_bins) if H_X[i][j][k] > 0 and H_Xbar[i][j][k] > 0])
		weights_Xbar = np.array([H_Xbar[i][j][k] for i in range(n_bins) for j in range(n_bins) for k in range(n_bins) if H_X[i][j][k] > 0 and H_Xbar[i][j][k] > 0])

		# Length of non-zero binned values
		N_binned = len(X_m2_binned)
		Nbar_binned = len(Xbar_m2_binned)

		return np.array(X_m2_binned), np.array(Xbar_m2_binned), weights_X, weights_Xbar, N_binned, Nbar_binned

	def generate_binned_dWq(self, n_perm, q, n_bins, min_max_dictionary = None, dWq_file_name = None, save = False):
		"""
		Generate and record binned delta Wq data 
		"""
		if dWq_file_name == None:
			dWq_file_name = rf'dWq_{q}_binned_{n_bins}.txt'
		dWq = []
		# The CPC distribution can be obtained in two ways
		if self.shuffle_or_master == 'master':
			# Generate the permutation data
			for i in tqdm(range(n_perm), ncols = 100, ascii = "░▒█"):
				# Grab a subset of the larger data file
				X_m2    = self.get_subset(self.X_data, self.N_len)
				Xbar_m2 = self.get_subset(self.Xbar_data, self.Nbar_len)

				# Get binned data
				X_m2_binned, Xbar_m2_binned, weights_X, weights_Xbar, N_binned, Nbar_binned = self.get_binned_subset(X_m2, Xbar_m2, n_bins, min_max_dictionary)
				
				# Compute delta W
				dWq_i = self.compute_dWq(q, X_m2_binned, Xbar_m2_binned, weights_X, weights_Xbar, N = N_binned, Nbar = Nbar_binned)
	
				# Write out the dW data
				if save == True:
					# Open a file and record the EMD_i data
					with open(dWq_file_name, 'a') as f:
						# Create the csv writer
						writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
						# Write EMD_i's
						writer.writerow(dWq_i)
				
				dWq.append(np.array(dWq_i))
	
			return np.array(dWq, dtype=object)
		else: # Shuffled case
			# Grab a subset of the larger data file
			X_m2    = self.get_subset(self.X_data, self.N_len)
			Xbar_m2 = self.get_subset(self.Xbar_data, self.Nbar_len)
			# Generate the permutation data
			for i in tqdm(range(n_perm), ncols = 100, ascii = "░▒█"):
				perm_min = 1
				if self.N_len < self.Nbar_len: perm_max = self.N 
				else: perm_max = self.Nbar_len
				
				# Shuffling procedure
				n_swap = random.randint(perm_min, perm_max)
				rand_A = random.sample(range(0, self.N_len), n_swap)
				rand_B = random.sample(range(0, self.Nbar_len), n_swap)
				
				# Get permuted datasets
				X_m2_perm, Xbar_m2_perm = swap(X_m2, Xbar_m2, rand_A, rand_B)
	
				# Get binned data
				X_m2_binned, Xbar_m2_binned, weights_X, weights_Xbar, N_binned, Nbar_binned = self.get_binned_subset(X_m2_perm, Xbar_m2_perm, n_bins, min_max_dictionary)
				
				# Compute delta W
				dWq_i = self.compute_dWq(q, X_m2_binned, Xbar_m2_binned, weights_X, weights_Xbar, N = N_binned, Nbar = Nbar_binned)
	
				# Write out the dW data
				if save == True:
					# Open a file and record the EMD_i data
					with open(dWq_file_name, 'a') as f:
						# Create the csv writer
						writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
						# Write EMD_i's
						writer.writerow(dWq_i)
				
				dWq.append(np.array(dWq_i))
	
			return np.array(dWq)

	def get_binned_dWq(self, n_perm, n_bins, min_max_dictionary = None, dWq_file_name = None, save = False):
		"""
		Generate binned dWq values for all input q values
		"""
		if type(self.q) == list:
			dWq = []
			for i in range(len(self.q)):
				print('q = ', self.q[i])
				dWq.append(self.generate_binned_dWq(n_perm, self.q[i], n_bins, min_max_dictionary, dWq_file_name[i], save))
			return np.array(dWq)
		else:
			return self.generate_binned_dWq(n_perm, self.q, n_bins, min_max_dictionary, dWq_file_name, save)

	#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
	#------------------------------------------------ Sliced Wasserstein Distance -------------------------------------------------#
	#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#

	def compute_SWq(self, X_data, Xbar_data, q, N_slices):
	   """
	   Calculates the sliced Wasserstein distance between two sets of data.
	   dist1: 	 numpy array of shape (n_samples, n_features)
	   dist2: 	 numpy array of shape (n_samples, n_features)
	   N_slices: int, number of projections to use for SWD
	   dim:      dimenson of the input data
	   
	   return: float, the sliced Wasserstein distance between dist1 and dist2
	   """
	   # Extract the dimensionality of the dataset
	   dim = X_data[0].shape[0]

	   # Create projection matrix i.e. generate random samples from the unit sphere
	   projections = np.asarray([w / np.sqrt((w**2).sum()) for w in (np.random.normal(size=(N_slices, dim)))])
	   
	   # Projection of the distribution
	   X_projection = X_data.dot(projections.T)
	   Xbar_projection = Xbar_data.dot(projections.T)

	   # Computing the Wasserstein distance from the two projected distribution i.e.the sorted difference
	   SWq = (1/self.mass**2)**q * np.power(np.abs(np.sort(X_projection.T, axis=1) - np.sort(Xbar_projection.T, axis=1)), q)
	   
	   return np.power(SWq.mean(), 1/q)


	def generate_SWq(self, n_perm, q, N_slices, SWq_file_name = None, save = False):
		SWq = []
		# We have two different methods of generating the CPC PDFs, shuffling CPV + CPC or by sampleing directly from two CPC datasets
		if self.shuffle_or_master == 'master':
			# Generate the permutation data
			for i in tqdm(range(n_perm), ncols = 100, ascii = "░▒█"):
			#for i in track(range(n_perm), disable = True):
				# Grab a subset of the larger data file
				X_m2_perm    = self.get_subset(self.X_data, self.N_len)
				Xbar_m2_perm = self.get_subset(self.Xbar_data, self.Nbar_len)
				
				# Compute delta SWq
				SWq_i = self.compute_SWq(X_m2_perm, Xbar_m2_perm, q, N_slices)
	
				# Write out the dW data
				if save == True:
					# Open a file and record the EMD_i data
					with open(SWq_file_name, 'a') as f:
						# Create the csv writer
						writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
						# Write dW's
						writer.writerow([SWq_i])
				SWq.append(np.array(SWq_i))
			return np.array(SWq)
		else:
			# Grab a subset of the larger data file
			X_m2    = self.get_subset(self.X_data, self.N_len)
			Xbar_m2 = self.get_subset(self.Xbar_data, self.Nbar_len)
			
			assert(not(any(i in X_m2 for i in Xbar_m2)))
			# Generate the permutation data from just the above to files
			for i in tqdm(range(n_perm), ncols = 100, ascii = "░▒█"):
				# Work with copies to avoid in place memory issues
				X_m2_copy = X_m2.copy()
				Xbar_m2_copy = Xbar_m2.copy()
				
				X_Xbar_m2_shuffle = np.vstack((X_m2_copy, Xbar_m2_copy))
				
				X_Xbar_m2_shuffle = np.random.permutation(X_Xbar_m2_shuffle)
				X_Xbar_m2_split = np.array_split(X_Xbar_m2_shuffle, [self.N_len], axis = 0)
				
				X_m2_perm = np.array(X_Xbar_m2_split[0])
				Xbar_m2_perm = np.array(X_Xbar_m2_split[1])
				
				# Compute SWq 
				SWq_i = self.compute_SWq(X_m2_perm, Xbar_m2_perm, q, N_slices)
				
				# Write out the dW data
				if save == True:
					# Open a file and record the EMD_i data
					with open(SWq_file_name, 'a') as f:
						# Create the csv writer
						writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
						# Write dW's
						writer.writerow(SWq_i)
				SWq.append(np.array(SWq_i))
			return np.array(SWq)
