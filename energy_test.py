"""
# energy_test.py is a part of the EMD4CPV package.
# Copyright (C) 2023 EMD4CPV authors (see AUTHORS for details).
# EMD4CPV is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

import pandas as pd
import os
import argparse
from scipy.stats import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
from subprocess import call

sns.set()
sns.set_style("ticks")
sns.set_context("paper", font_scale = 3.0)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 25

class cd:
	"""
	Context manager for changing the current working directory
	"""
	def __init__(self, newPath):
		self.newPath = os.path.expanduser(newPath)

	def __enter__(self):
		self.savedPath = os.getcwd()
		os.chdir(self.newPath)

	def __exit__(self, etype, value, traceback):
		os.chdir(self.savedPath)

def compute_T(X_m2, Xbar_m2, sigma):
	"""
	Simple algorithm for computing the energy test statistic

	X_m2:    X-particle Dalitz data
	Xbar_m2: Xbar-particle Dalitz data
	sigma:   tunable parameters of the energy test
	"""
	TT, TbarTbar,TTbar = [],[],[]

	# Define number of events for normalization
	N    = len(X_m2)
	Nbar = len(Xbar_m2)

	# Compute the cost matrices
	cost = cdist(X_m2, Xbar_m2, 'euclidean')
	cost_B0 = cdist(X_m2, X_m2, 'euclidean')
	cost_B0bar = cdist(Xbar_m2, Xbar_m2, 'euclidean')

	for i in range(N):
		for j in range(i+1,N):
			TT.append(np.exp(- 0.5 * (cost_B0[i][j]**2 / sigma**2)))

	for i in range(Nbar):
		for j in range(i+1,Nbar):
			TbarTbar.append(np.exp(- 0.5 * (cost_B0bar[i][j]**2 / sigma**2)))

	for i in range(N):
		for j in range(Nbar):
			TTbar.append(np.exp(- 0.5 * (cost[i][j]**2 / sigma**2)))

	T = (1/(N*(N-1))) * sum( TT ) + (1/(Nbar*(Nbar-1))) * sum( TbarTbar ) - (1/(N*Nbar)) * sum( TTbar )

	return T

def compute_sparse_T(X_m2, Xbar_m2, sigma):
	"""
	Same computation of T but without the need for the full form of the cost matrix
	"""
	TT, TbarTbar,TTbar = [],[],[]

	# Define number of events for normalization
	N    = len(X_m2)
	Nbar = len(Xbar_m2)

	for i in range(N):
		for j in range(i+1,N):
			dist = cdist([X_m2[i]], [X_m2[j]])
			TT.append(np.exp(- 0.5 * (dist**2 / sigma**2)))

	for i in range(Nbar):
		for j in range(i+1,Nbar):
			dist = cdist([Xbar_m2[i]], [Xbar_m2[j]])
			TbarTbar.append(np.exp(- 0.5 * (dist**2 / sigma**2)))

	for i in range(N):
		for j in range(Nbar):
			dist = cdist([X_m2[i]], [Xbar_m2[j]])
			TTbar.append(np.exp(- 0.5 * (dist**2 / sigma**2)))


	T = (1/(N*(N-1))) * sum( TT ) + (1/(Nbar*(Nbar-1))) * sum( TbarTbar ) - (1/(N*Nbar)) * sum( TTbar )

	return T

def compute_T_w_manet(X_m2, Xbar_m2, path_to_manet):
	"""
	Again, same computation of the energy test statistic but now interfaced
	with Manet for computation acceleration is required. Useful for generating
	CP conserving PDF data

	Note: You will need to modify energy_test_w_manet.sh with the relevant sigma
	parameter as well as path to your data folder. Additionally you will need to 
	copy this file into the bin directory within your Manet path (Manet/bin/)
	"""
	# Open a file to be used as a txt file for Manet
	with open(rf'data/X_m2_temp.txt', 'w', encoding="utf-8") as f:
		writer = csv.writer(f, delimiter=' ')
		writer.writerows(X_m2)

	# Open a file to be used as a txt file for Manet
	with open(rf'data/Xbar_m2_temp.txt', 'w', encoding="utf-8") as f:
		writer = csv.writer(f, delimiter=' ')
		writer.writerows(Xbar_m2)

	with cd(path_to_manet + '/bin'):
		os.chmod(path_to_manet + '/bin/energy_test_w_manet.sh', 0o755)
		rc = call(path_to_manet + '/data/home/menzoad/Manet/bin/energy_test_w_manet.sh', shell=True)

	# Read T value from file produced by Manet
	with open(r'data/T_w_manet.txt', 'r') as csvfile:
		csvreader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
		for row in csvreader:
			T = row

	return T[0]
