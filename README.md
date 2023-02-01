# EMD4CPV - [arXiv:2301.13211](https://arxiv.org/abs/2301.13211)

Requirements: The ``delta_Wq_versus`` subclass requires Python 3.10+ while all other classes require Python 3+.

The program architecture is hierarchically structured, resembling a nested doll of classes and subclasses. ``delta_Wq`` is the highest level class and contains the sub-class ``delta_Wq_statistics``, which in turn contains the sub-class ``delta_Wq_fit``,  which finally contains the sub-class ``delta_Wq_versus``. Typical usage of the library will follow a nested call of functions within each class. Because each sub-class inherits all functions from the previous class this allows the user to work at any level of the program architecture while only needing to initialize one class instance. While the use case of the program is oriented toward 3--body decays, the code is generic enough such that it can be easily generalized and used with any n-dimensional dataset. 

Below we summarize briefly the software pipeline (see the documentation as well as the example Python notebook ``EMD_CPV_example.ipynb`` within the repository for more details):

1. The  ``delta_Wq`` class contains functions which allow the user to input two n-dimensional distributions and obtain the associated binned or unbinned delta W_q values chosen by the optimization. Since in most cases the CP conserving distributions (functionals of delta W_q)  need to be calculated, the class is set up such that the generation of the CP even distributions via the master or permutation methods can be done efficiently by randomly selecting a subset of unique datapoints from a larger datapool provided by the user in the form of a text file. In addition, this class may be used to compute the sliced Wasserstein distance SW_q.

2. Once the delta W_q ensemble is obtained, the subclass ``delta_Wq_statistics`` can be used to compute the W_q, I_q, or any other user defined statistical distributions. 

3. Oftentimes, when computing the p-values from the CPV datasets a fit is needed in order to extrapolate outside the ranges of explicitly calculated CP conserving distributions. These fits can be performed using the ``delta_Wq`` subclass which allows the user to iteratively fit to any distribution within the SciPy.stats library and return the associated PDFs, CDFs, SFs, chi^2-values, as well as the PDFs, CDFs, and SFs associated with the +/- 1-sigma errors on the fit parameters. 

4. Finally, the ``delta_Wq_versus`` subclass may be used to iteratively compare the sensitivity of different statistics on ensembles of like datasets. 

5. Additionally, for convenience, the script ``energy_test.py`` provides a Python implementation of the energy test statistic, i.e., the computation of the test statistic T between two n-dimensional distributions for use when computing CPV statistic values in ``delta_Wq_versus``.
