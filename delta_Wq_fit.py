"""
# delta_Wq_fit.py is a part of the EMD4CPV package.
# Copyright (C) 2023 EMD4CPV authors (see AUTHORS for details).
# EMD4CPV is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

from delta_Wq_statistics import *
import scipy.stats
from scipy.optimize import curve_fit
from scipy.stats import chisquare

class delta_Wq_fit(delta_Wq_statistics):
	"""
	This class provides fitting functionality to the test statistics computed in delta_Wq_statistics.py.

	The statistics can be fit to any continuour distribution found in Scipy.stats. Additionally, 1-sigma
	errors on all ffit parameters can be output if desired.
	"""
	def distribution_data(self, statistic):
		"""
		Returns many useful distributions properties

		statistic (numpy array): List of statistic data to be fit

		Returns bin_centers, count data, cdf data, sf data, as well as the minimum and maximum
		"""
		min_stat, max_stat = min(statistic), max(statistic)
		counts, bin_edges = np.histogram(statistic, bins = 50, density = True)
		bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
		statistic_cdf = [(counts[i]+sum(counts[:i]))/sum(counts) for i in range(len(counts))]
		statistic_sf = [(1-statistic_cdf[i]) for i in range(len(statistic_cdf))]

		return bin_centers, counts, statistic_cdf, statistic_sf, min_stat, max_stat

	def load_statistic_data(self, path):
		"""
		Load statistic data from file

		path (string): path to statistic data

		Returns numpy array with statistic data
		"""
		statistic=[]
		# Read data from path
		with open(path, 'r') as csvfile:
			    # Create csv reader object
			    csvreader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC, delimiter = ' ')
			    # Extract row by row
			    for row in csvreader:
			        statistic.append(row)
		return np.array(statistic).flatten()

	def fit_dist(self, dist_key, data):
		"""
		Generic fitting function which avoids the need for all of the distributions within the class, now any 
		continuous function within SciPy.stats can be fit. 

		dist_key (string): Name of the distribution to fit following the naming conventions within SciPy

		data (list): ----- List of collected statistic data

		Returns fit results including lambda functions of the pdf, cdf, and sf as well as the chi^2 estimate for the fit
		"""
		dist = getattr(scipy.stats, dist_key)
		params = dist.fit(data)
		arg = params[:-2]
		loc = params[-2]
		scale = params[-1]
		if arg:
		    dist_pdf = lambda x: dist.pdf(x, *arg, loc = loc, scale = scale)
		    dist_cdf = lambda x: dist.cdf(x, *arg, loc = loc, scale = scale)
		    dist_sf  = lambda x: dist.sf(x, *arg, loc = loc, scale = scale)
		else:
		    dist_pdf = lambda x: dist.pdf(x, loc = loc, scale = scale)
		    dist_cdf = lambda x: dist.cdf(x, loc = loc, scale = scale)
		    dist_sf  = lambda x: dist.sf(x, loc = loc, scale = scale)
		
		# Compute the chi^2
		counts, edges = np.histogram(data, bins = 50)
		n_bins = len(edges) - 1
		observed_values = counts
		if arg: 
			cdf = dist.cdf(edges, *arg, loc = loc, scale = scale)
		else:
			cdf = dist.cdf(edges, loc = loc, scale = scale)
		expected_values = len(data) * (np.diff(cdf)/np.diff(cdf).sum())
		chi2 , p = chisquare(observed_values, expected_values, ddof=len(arg)+2)
		
		return params, dist_pdf, dist_cdf, dist_sf, chi2

	def fit_dist_w_error(self, dist_key, data, n_bins = None):
		"""
		This function should typically used after determining the best fit distribution to use from fit_dist.
		Once determined this function will optimize the fit with the assignment of Poisson-like arror bars on the 
		bin counts and in addition return the covariance matrix of the fit, allowing for the extraction of 
		fit parameter errors. These errors may then be used to obtain an estimated error on any p-value determination

		dist_key (string): Name of the distribution to fit following the naming conventions within SciPy
		
		data (list): ----- List of collected statistic data

		Returns fit results including lambda functions of the pdf, cdf, and sf as well as the chi^2 estimate for the fit
		"""
		# Get the specified distribution fit parameters
		#print('dist_key: ', dist_key)
		dist = getattr(scipy.stats, dist_key)
		# Get inital guess
		params, dist_pdf, dist_cdf, dist_sf, chi2 = self.fit_dist(dist_key, data)
		arg = params[:-2]
		loc = params[-2]
		scale = params[-1]
		#print('Initial param guess: ', list(params[::]))

		# Estimate the initial errors in the PDF
		if n_bins == None:
			n_bins = 50
			term = False
			# Loop until fit and errors are reasonable
			while term != True: 
				#print('n_bins = ', n_bins)
				counts, bins = np.histogram(data, bins = n_bins)#, density = True)
				statistic_cdf = [(counts[i]+sum(counts[:i]))/sum(counts) for i in range(len(counts))]
				statistic_sf = [(1-statistic_cdf[i]) for i in range(len(statistic_cdf))]
				# Propagate the errors for the PDF
				std = np.sqrt(counts * (1 - (counts / len(data))))
				# Start with the sum 
				delta_sum = np.sqrt(np.sum(std**2))
				# Error of the division
				delta_density = np.sqrt( (std / counts)**2 + (delta_sum / np.sum(counts))**2)
				# Get the density count data
				counts, bins = np.histogram(data, bins = n_bins, density = True)
				# Get the error of the density distribution
				std_pdf = counts * delta_density
				# Define PDF
				dist_pdf_opt = lambda x, *params: dist.pdf(x, *params)
				# Get bin centers
				bin_centers = 0.5 * (bins[1:] + bins[:-1])
				try:
					params_opt, cov = curve_fit(dist_pdf_opt, bin_centers, counts, p0 = list(params[::]), sigma = std_pdf, absolute_sigma = True, maxfev = 5000)#, sigma = std)
					# Compute the difference
					params_diff = np.abs(params_opt - params)
					# Capture the cases where the fit has diverged or not changed at all
					assert np.sum(params_diff) <= 50.0 and np.sum(params_diff) > 0.0
					#print('Optimal params: ', params_opt)
					#print('Initial - Optimal: ', params_diff)
					
					# Redefine the pdf, cdf, sf with the optimal parameters
					dist_pdf = lambda x: dist.pdf(x, *params_opt)
					dist_cdf = lambda x: dist.cdf(x, *params_opt)
					dist_sf = lambda x: dist.sf(x, *params_opt)
			
					# The errors are given by the sqrt of the diagnoal elements of the covariance matrix
					params_err = np.sqrt(np.diag(cov))
					#print('Param 1-sigma errors: ', params_err)
					# Get the +/- nstd fit parameters 
					nstd = 1. 
					pars_plus = params_opt + nstd * params_err
					pars_minus = params_opt - nstd * params_err
			
					# Create lambda functions for the upper and lower error distributions
					dist_pdf_plus = lambda x: dist.pdf(x, *pars_plus)
					dist_cdf_plus = lambda x: dist.cdf(x, *pars_plus)
					dist_sf_plus  = lambda x: dist.sf(x, *pars_plus)
					dist_pdf_minus = lambda x: dist.pdf(x, *pars_minus)
					dist_cdf_minus = lambda x: dist.cdf(x, *pars_minus)
					dist_sf_minus  = lambda x: dist.sf(x, *pars_minus)
				
					#print('Final params: ', params)
					#print('Covariance matrix: ', cov)
					term = True
				except:
					# If the fit didn't work, reduce the binning and try again
					n_bins = n_bins - 1
		return dist_pdf, dist_pdf_plus, dist_pdf_minus, dist_cdf, dist_cdf_plus, dist_cdf_minus, dist_sf, dist_sf_plus, dist_sf_minus

	
	def fit_statistic(self, statistic_key, statistic_data, fit_distribution_keys, plot_details = False):
		"""
		This function allows the user to input a list of fit distributions to be fit on input statistic data.

		statistic_key (string): ---------------- Name of statistic to be fit

		statistic_data (string or numpy array):  Path to statistic data or list of statistic data. If path is provided
												 the data must be formatted in the conventions of load_statistic_data

		fit_distribution_keys (list of strings): List containing strings of distributions to be fit. Names of the
												 distributions follow the conventions of scipy.stats

		plot_details (bool): ------------------- Plot pdf, cdf, and sf for all fit distributions

		returns dictionaries of fit data including 
		"""
		# Load statistic data
		if type(statistic_data) == str:
			statistic = self.load_statistic_data(statistic_data)
		else:
			statistic = statistic_data
		fit_data = {'pdf':[], 'cdf':[], 'sf':[], 'chi2':[]}
		for keys in fit_distribution_keys:
			print(keys)
			params, pdf, cdf, sf, chi2 = self.fit_dist(keys, statistic)
			fit_data['pdf'].append(pdf)
			fit_data['cdf'].append(cdf)
			fit_data['sf'].append(sf)
			fit_data['chi2'].append(chi2)

		bin_centers, counts, statistic_cdf, statistic_sf, min_stat, max_stat = self.distribution_data(statistic)
		if plot_details == True:
			x = np.linspace(min_stat - np.abs((0.00001 * max_stat))/max_stat, max_stat + np.abs((0.00001 * max_stat))/max_stat, 500)
		
			fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize = (11,11), sharex = True)
			if  statistic_key.strip() == 'Wq':
				ax1.set_title(rf'$q = {self.q}$', pad = 10)
				ax3.set_xlabel(r'$W_q$')
			elif statistic_key.strip() == 'log_IWq' or statistic_key.strip() == 'IWq':
				ax1.set_title(rf'$q = {self.q}$', pad = 10)
				ax3.set_xlabel(r'$I_q$')
			elif statistic_key.strip() == 'SWq':
				ax1.set_title(rf'$q = {self.q}$', pad = 10)
				ax3.set_xlabel(r'$SW_q$')
			ax1.plot(bin_centers, counts, 'o')
			ax2.plot(bin_centers, statistic_cdf, 'o')
			ax3.plot(bin_centers, statistic_sf, 'o')
			for i in range(len(fit_distribution_keys)):
				print(fit_distribution_keys[i].strip("'"), 'chi2 =', fit_data['chi2'][i])
				ax1.plot(x, fit_data['pdf'][i](x), label=rf'$\texttt{{{fit_distribution_keys[i].strip()}}}$ $(\chi^2 = {fit_data["chi2"][i]:.2f})$')
				ax2.plot(x, fit_data['cdf'][i](x), label=rf'$\texttt{{{fit_distribution_keys[i].strip()}}}$ $(\chi^2 = {fit_data["chi2"][i]:.2f})$')
				ax3.plot(x, fit_data['sf'][i](x), label=rf'$\mathrm{{{fit_distribution_keys[i].strip()}}}$')

			ax1.set_ylabel(r"$\mathrm{PDF}$")
			ax2.set_ylabel(r"$\mathrm{CDF}$")
			ax3.set_ylabel(r"$\mathrm{SF}$")
			ax1.set_xlim(x[0], x[-1])
			ax2.set_xlim(x[0], x[-1])
			ax3.set_xlim(x[0], x[-1])
			ax3.set_yscale('log')
			ax2.legend(frameon=False, fontsize='xx-small')
			fig.tight_layout()
			fig.savefig('plots/' + statistic_data.replace(".txt", "").replace("data/","") + '_pdf_cdf_sf_fit_' + self.shuffle_or_master + '.pdf', dpi = 300, pad_inches = .1, bbox_inches = 'tight')
		return fit_data, bin_centers, counts, statistic_cdf, statistic_sf, min_stat, max_stat

	def get_fit_sf(self, statistic_path, fit_key):
		"""
		Get the survival function for the fit distribution

		statistic_path (string): path to statistic data

		fit_key (string): name of the distribution to be fit and sf returned

		Returns lambda function of fit distribution survival function
		"""
		statistic = self.load_statistic_data(statistic_path)
		params, dist_pdf, dist_cdf, dist_sf, chi2 = self.fit_dist(fit_key, statistic)
		return dist_sf

	def get_fit_sf_w_error(self, statistic_path, fit_key):
		"""
		Get the survival function with one-sigma confidence bands for the fit distribution

		statistic_path (string): path to statistic data

		fit_key (string): name of the distribution to be fit and sf returned

		Returns lambda function of fit distribution survival function, as well as upper and lower 
		fit distribution representing the one-sigma fit parameter errors
		"""
		print('staistic path: ', statistic_path)
		statistic = self.load_statistic_data(statistic_path)
		pdf, pdf_plus, pdf_minus, cdf, cdf_plus, cdf_minus, sf, sf_plus, sf_minus = self.fit_dist_w_error(fit_key, statistic)
		return sf, sf_plus, sf_minus