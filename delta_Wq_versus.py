"""
# delta_Wq_versus.py is a part of the EMD4CPV package.
# Copyright (C) 2023 EMD4CPV authors (see AUTHORS for details).
# EMD4CPV is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

from delta_Wq_fit import *
from energy_test import compute_T, compute_sparse_T, compute_T_w_manet
from matplotlib.offsetbox import AnchoredText
import re
from cycler import cycler
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y'])))

class delta_Wq_versus(delta_Wq_fit):
	"""
	This class allows for a CPV sensitivity comparison between test statistics on like-datasets.
	"""
	def __init__(self, X_data, Xbar_data, N_len, Nbar_len, q, mass, pool_len, shuffle_or_master, statistic_fit_dictionary):
		self.statistic_fit_dictionary = statistic_fit_dictionary
		super().__init__(X_data, Xbar_data, N_len, Nbar_len, q, mass, pool_len, shuffle_or_master)

	def battle(self, n_battles, statistic_manual, statistic_paths, min_max_dictionary = None):
		"""
		Using the fit survival functions, this function computes the p-value for each statistic and returns
		the p-values and associated performance of each statistic.

		n_battles (int):                 number of like-dataset comparisons.
		statistic_manual (dictionary):   this dictionary contains the chosen fit distribution to be used when determining p-values from the SF.
		statistic_paths (dictionary):    this dictionary contains the relevant paths to the test statistic data (to be fit and SF extracted)
		min_max_dictionary (dictionary): global min-maxes for a given Dalitz plot distribution (only requred for the binned Wq)
		"""
		# Get statistic keys from statistic versus list
		keys = statistic_manual.keys()
		print('keys: ', keys)
		contenders = {}
		# Get the fit survial functions
		sfs = {}
		for key in keys:
			match key:
				case 'T':
					contenders[key] = key
					# Get survival distribution lambda functions
					sf, sf_plus, sf_minus = self.get_fit_sf_w_error(statistic_paths[key], self.statistic_fit_dictionary[key])
					sfs[key] = [sf, sf_plus, sf_minus]
				case 'Wq':
					contenders[key] = []
					for i in range(len(statistic_manual[key])):
						# Get survival distribution lambda functions
						sf, sf_plus, sf_minus = self.get_fit_sf_w_error(statistic_paths[key][i], self.statistic_fit_dictionary[key][i])
						Wq_key = statistic_paths[key][i].replace('_data.txt','').replace('data/','')
						contenders[key].append(Wq_key)
						sfs[Wq_key] = [sf, sf_plus, sf_minus]
				case 'IWq':
					contenders[key] = []
					for i in range(len(statistic_manual[key])):
						# Get survival distribution lambda functions
						sf, sf_plus, sf_minus = self.get_fit_sf_w_error(statistic_paths[key][i], self.statistic_fit_dictionary[key][i])
						IWq_key = statistic_paths[key][i].replace('_data.txt','').replace('data/','')
						contenders[key].append(IWq_key)
						sfs[IWq_key] = [sf, sf_plus, sf_minus]
				case 'binned_Wq':
					contenders[key] = []
					for i in range(len(statistic_manual[key])):
						# Get survival distribution lambda functions
						sf, sf_plus, sf_minus = self.get_fit_sf_w_error(statistic_paths[key][i], self.statistic_fit_dictionary[key][i])
						Wq_key = statistic_paths[key][i].replace('_data.txt','').replace('data/','')
						contenders[key].append(Wq_key)
						sfs[Wq_key] = [sf, sf_plus, sf_minus]
				case 'log_IWq':
					contenders[key] = []
					for i in range(len(statistic_manual[key])):
						sf, sf_plus, sf_minus = self.get_fit_sf_w_error(statistic_paths[key][i], self.statistic_fit_dictionary[key][i])
						IWq_key = statistic_paths[key][i].replace('_data.txt','').replace('data/','')
						print(IWq_key)
						contenders[key].append(IWq_key)
						sfs[IWq_key] = [sf, sf_plus, sf_minus]
				case 'SWq':
					contenders[key] = []
					for i in range(len(statistic_manual[key])):
						sf, sf_plus, sf_minus = self.get_fit_sf_w_error(statistic_paths[key][i], self.statistic_fit_dictionary[key][i])
						SWq_key = statistic_paths[key][i].replace('_data.txt','').replace('data/','')
						print(SWq_key)
						contenders[key].append(SWq_key)
						sfs[SWq_key] = [sf, sf_plus, sf_minus]

		# Initialize results array
		print('Contenders: ', contenders)
		p = []
		for i in tqdm(range(n_battles), ncols = 100, ascii = "░▒█"):
			battle_i, p_i = [],[]
			# Initialize m2 event data to be compared
			X_m2    = self.get_subset(self.X_data, self.N_len)
			Xbar_m2 = self.get_subset(self.Xbar_data, self.Nbar_len)
			# Define the weights
			weights_X = np.ones(self.N_len)*1./float(self.N_len)
			weights_Xbar = np.ones(self.Nbar_len)*1./float(self.Nbar_len)
		
			for key in keys:
				match key:
					case 'T':
						T = compute_T(X_m2, Xbar_m2, *statistic_manual[key])
						# Other methods to compute T. See energy_test.py for more details.
						#T = compute_sparse_T(X_m2, Xbar_m2, *statistic_manual[key])
						#T = compute_T_w_manet(X_m2, Xbar_m2, *statistic_manual[key])
						p_i.append([sfs[key][0](T), np.abs(sfs[key][1](T) - sfs[key][0](T)), np.abs(sfs[key][2](T) - sfs[key][0](T))])

					case 'Wq':
						for i in range(len(statistic_manual[key])):
							# Generate dWq data
							dWq = self.compute_dWq(statistic_manual[key][i][0], X_m2, Xbar_m2, weights_X, weights_Xbar)
							# Get nominal value from the event datasets in question
							statistic_value = self.get_statistic(dWq, key, statistic_manual[key][i])
							# Collect p-value information from each survival function
							p_nominal = sfs[contenders[key][i]][0](statistic_value)
							# Account for the fact that sfs_plus might not always give the upper errorbar
							if sfs[contenders[key][i]][1](statistic_value) > p_nominal: 
								p_error_plus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							else:
								p_error_plus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							p_i.append([p_nominal, p_error_plus, p_error_minus])
					
					case 'IWq':
						for i in range(len(statistic_manual[key])):
							# Generate dWq data
							dWq = self.compute_dWq(statistic_manual[key][i][0], X_m2, Xbar_m2, weights_X, weights_Xbar)
							# Get nominal value from the event datasets in question
							statistic_value = self.get_statistic(dWq, key, statistic_manual[key][i])
							# Collect p-value information from each survival function
							p_nominal = sfs[contenders[key][i]][0](statistic_value)
							# Account for the fact that sfs_plus might not always give the upper errorbar
							if sfs[contenders[key][i]][1](statistic_value) > p_nominal: 
								p_error_plus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							else:
								p_error_plus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							p_i.append([p_nominal, p_error_plus, p_error_minus])
					
					case 'binned_Wq':
						for i in range(len(statistic_manual[key])):
							# Get the binned data
							X_m2_binned, Xbar_m2_binned, weights_X, weights_Xbar, N_binned, Nbar_binned = self.get_binned_subset(X_m2, Xbar_m2, n_bins = 50, min_max_dictionary = min_max_dictionary)
							# Generate dWq data
							dWq = self.compute_dWq(statistic_manual[key][i][0], X_m2_binned, Xbar_m2_binned, weights_X, weights_Xbar, N = N_binned, Nbar = Nbar_binned)
							# Get nominal value from the event datasets in question
							statistic_value = self.get_statistic(dWq, 'Wq', statistic_manual[key][i])
							# Compute p-value and append to record
							p_nominal = sfs[contenders[key][i]][0](statistic_value)
							# Account for the fact that sfs_plus might not always give the upper errorbar
							if sfs[contenders[key][i]][1](statistic_value) > p_nominal: 
								p_error_plus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							else:
								p_error_plus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							p_i.append([p_nominal, p_error_plus, p_error_minus])

					case 'log_IWq':
						for i in range(len(statistic_manual[key])):
							# Generate dWq data
							dWq = self.compute_dWq(statistic_manual[key][i][0], X_m2, Xbar_m2, weights_X, weights_Xbar)
							# Get nominal value from the event datasets in question
							statistic_value = self.get_statistic(dWq, key, statistic_manual[key][i])
							# Collect p-value information from each survival function
							p_nominal = sfs[contenders[key][i]][0](statistic_value)
							# Account for the fact that sfs_plus might not always give the upper errorbar
							if sfs[contenders[key][i]][1](statistic_value) > p_nominal: 
								p_error_plus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							else:
								p_error_plus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							p_i.append([p_nominal, p_error_plus, p_error_minus])

					case 'SWq':
						for i in range(len(statistic_manual[key])):
							statistic_value = self.compute_SWq(X_m2, Xbar_m2, *statistic_manual[key][i])
							# Collect p-value information from each survival function
							p_nominal = sfs[contenders[key][i]][0](statistic_value)
							# Account for the fact that sfs_plus might not always give the upper errorbar
							if sfs[contenders[key][i]][1](statistic_value) > p_nominal: 
								p_error_plus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							else:
								p_error_plus = np.abs(sfs[contenders[key][i]][2](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
								p_error_minus = np.abs(sfs[contenders[key][i]][1](statistic_value) - sfs[contenders[key][i]][0](statistic_value))
							p_i.append([p_nominal, p_error_plus, p_error_minus])
			
			# Append the p-value data for this dataset
			p.append(np.array(p_i))

		# Get performance data
		contenders_list = list(sfs.keys())
		n_contenders = len(sfs)
		print(n_contenders)
		wins = np.zeros((n_contenders, n_contenders))
		for i in range(len(p)):
			for j in range(n_contenders - 1):
				for k in range(j+1, n_contenders):
					if p[i][j][0] < p[i][k][0]: wins[j,k] += 1

		# Only interested in the upper triangular piece + total win percentage
		win_percentage = np.array(sum((wins[i,i+1:].tolist() for i in range(n_contenders)), [])) / n_battles 
		print(win_percentage)

		return np.array(p), win_percentage, contenders_list

	def plot_battle(self, n_battles, statistic_manual, statistic_paths, min_max_dictionary = None, save_p_vals = False, plot_w_errors = False, label = None):
		# Get p-value, battle performance, and contender data
		p, win_percentage, contenders = self.battle(n_battles, statistic_manual, statistic_paths, min_max_dictionary)
		n_contenders = len(contenders)
		counter = 0

		contender_labels = []
		# Get fancy axis labels
		for contender in contenders:
			print(contender.split())
			if contender == 'T':
				contender_labels.append(r'$p(T)$')
			elif re.match("^W", contender):
			    print(contender.split('_'))
			    # Extract q value
			    contender_list = contender.split('_')
			    contender_labels.append(rf'$p(W_{{{contender_list[1]}}})$')
			elif re.match("^l", contender):
				print(contender.split('_'))
				# Extract q value
				contender_list = contender.split('_')
				contender_labels.append(rf'$p(I_{{{contender_list[2]}}})$')
			elif re.match("^S", contender):
				print(contender.split('_'))
				# Extract q value
				contender_list = contender.split('_')
				contender_labels.append(rf'$p(SW_{{{contender_list[2]}}})$')
			else:
				contender_labels.append(r'$p(W_q)$')
		print(contender_labels)

		# Compute the averages 
		avg_p, avg_p_errs = [],[]
		for i in range(len(contenders)):
			# Compute the average p-values
			p_contender = [p[j][i][0] for j in range(n_battles)]
			avg_p.append(np.average(p_contender))
			# Compute the upper and lower bounds on the average p-value
			p_err_plus = np.array([p[j][i][1] for j in range(n_battles)])
			avg_p_err_plus = np.sqrt(np.sum(p_err_plus**2)) / n_battles
			p_err_minus = np.array([p[j][i][2] for j in range(n_battles)])
			avg_p_err_minus = np.sqrt(np.sum(p_err_minus**2)) / n_battles
			avg_p_errs.append([avg_p_err_plus, avg_p_err_minus])

		# Algorithmically plot the versus results
		cmap = plt.get_cmap("tab10")
		for i in range(n_contenders - 1):
			for j in range(i+1, n_contenders):
				fig, ax = plt.subplots(1, 1, figsize=(10,6))
				# Plot on log scale for convinient viewing
				ax.set_yscale('log')
				ax.set_xscale('log')
				ax.set_aspect('equal')

				print(contenders[i], ' vs ', contenders[j])
				# Write out the win percentage of the x-axis
				at = AnchoredText(rf'$\epsilon = {win_percentage[counter]:.3f}$', prop=dict(size=16), frameon=False, loc='upper left')
				ax.add_artist(at)
				# Plot the results
				for k in range(len(p)):
					if plot_w_errors:
						ax.errorbar(p[k,i,0], p[k,j,0], yerr = [[p[k,j,2]], [p[k,j,1]]], xerr = [[p[k,i,2]], [p[k,i,1]]], fmt='o', capsize = 1, c = cmap(k%10))
					else: ax.plot(p[k][i][0], p[k][j][0], marker = 'o', linestyle='None', c=cmap(k%10)) 
				
				ax.set_xlabel(rf'{contender_labels[i].strip()}')
				ax.set_ylabel(rf'{contender_labels[j].strip()}')
				ax.axvline(x = avg_p[i], ls = '--', color = 'gray', alpha = 0.5)#, label = rf'$\mathrm{{{contenders[i].strip()}}}$ $\bar{{p}}={avg_p[i]:.2f}$')
				ax.axhline(y = avg_p[j], ls = '--', color = 'gray', alpha = 0.5)#, label = rf'$\mathrm{{{contenders[j].strip()}}}$ $\bar{{p}}={avg_p[j]:.2f}$')
				ax.axvspan(avg_p[i] - avg_p_errs[i][1], avg_p[i] + avg_p_errs[i][0], color='gray', alpha=0.3, lw = 0)
				ax.axhspan(avg_p[j] - avg_p_errs[j][1], avg_p[j] + avg_p_errs[j][0], color='gray', alpha=0.3, lw = 0)
				
				ax.set_xlim(float('10e-5'), 2.0)
				ax.set_ylim(float('10e-5'), 2.0)

				# Plot diagonal line to indicate the better p-value
				lims_a = [
				    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
				    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
				]
				ax.plot(lims_a, lims_a, 'k-', zorder=0)
				fig.tight_layout()
				if label == None:
					fig.savefig(rf'plots/{contenders[i]}_vs_{contenders[j]}_battles_{n_battles}_w_errors.pdf', dpi = 300, pad_inches = .1, bbox_inches = 'tight')
				else:
					fig.savefig(rf'plots/{contenders[i]}_vs_{contenders[j]}_battles_{n_battles}_w_errors_{label}.pdf', dpi = 300, pad_inches = .1, bbox_inches = 'tight')
				counter += 1

		# Save p-value data
		if save_p_vals == True:
			for i in range(len(contenders)):
				if label == None: file_name = rf'data/{contenders[i]}_p.txt'
				else: file_name = rf'data/{contenders[i]}_p_{label}.txt'
					
				# Open the file in the write mode
				with open(file_name, 'w') as f:
					print('Writing ', file_name)
					# Create the csv writer
					writer = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC, delimiter=' ')
					for j in range(len(p)):
						# Write p data
						writer.writerow([p[j][i][0], p[j][i][1], p[j][i][2]])

		return fig
