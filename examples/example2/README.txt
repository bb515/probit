example.py is for the GP hyperparameter, varphi is a single constant across all classes. The .png files are for this example.
general_example.py is for the GP hyperparameters, varphi is a (K, D) array, one for each class for each dimension of the data.


number of gibbs samples = 15 
	Samping monte carlo random variables one at a time over a for-loop

		201101342 function calls (201070089 primitive calls) in 313.126 seconds
	   Ordered by: cumulative time
	   List reduced from 2084 to 104 due to restriction <0.05>
	   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
	      400    1.199    0.003  307.396    0.768 /home/ben/multivar`iate_regression_GP_gibbs/probit/samplers.py:270(predict)
	     6000   16.118    0.003  304.171    0.051 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:158(expectation_wrt_u)
	  1200000   55.262    0.000  153.209    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:1777(cdf)
	  1200000    3.882    0.000  114.949    0.000 /home/ben/multivariate_regression_GP_gibbs/probit/utilities.py:37(sample_U)
	  1218000   22.400    0.000   94.172    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:931(rvs)

	After taking all monte carlo random variables at once (Optimisation 1 = 30 times speedup)
		 6504295 function calls (6473070 primitive calls) in 10.013 seconds
	   Ordered by: cumulative time
	   List reduced from 2084 to 104 due to restriction <0.05>
	   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
	      400    0.492    0.001    4.433    0.011 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:273(predict)
		2    0.122    0.061    4.075    2.037 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:114(sample)
	    21343    0.034    0.000    3.734    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_multivariate.py:635(rvs)
	    21343    0.629    0.000    3.567    0.000 {method 'multivariate_normal' of 'mtrand.RandomState' objects}
	     6000    0.505    0.000    2.513    0.000 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:167(expectation_wrt_u)

	After vectorising over X_new (as opposed to calculating x_new in a loop). (Optimisation 2 = further <2 times speedup)
		 3931593 function calls (3901778 primitive calls) in 6.528 seconds
	   Ordered by: cumulative time
	   List reduced from 2044 to 102 due to restriction <0.05>
	   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
		2    0.133    0.066    4.252    2.126 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:113(sample)
	    21492    0.035    0.000    3.887    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_multivariate.py:635(rvs)
	    21492    0.660    0.000    3.715    0.000 {method 'multivariate_normal' of 'mtrand.RandomState' objects}
	    21492    0.926    0.000    2.072    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/linalg/decomp_svd.py:16(svd)
		5    0.000    0.000    1.057    0.211 /home/ben/anaconda3/lib/python3.7/site-packages/matplotlib/pyplot.py:252(show)
		5    0.000    0.000    1.057    0.211 /snap/pycharm-professional/228/plugins/python/helpers/pycharm_matplotlib_backend/backend_interagg.py:21(__call__)
		5    0.000    0.000    1.056    0.211 /snap/pycharm-professional/228/plugins/python/helpers/pycharm_matplotlib_backend/backend_interagg.py:111(show)
		5    0.055    0.011    1.054    0.211 /snap/pycharm-professional/228/plugins/python/helpers/pycharm_matplotlib_backend/backend_interagg.py:67(show)
		1    0.025    0.025    1.003    1.003 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:208(predict_vector)
	       15    0.091    0.006    0.962    0.064 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:178(multidimensional_expectation_wrt_u)
	    21492    0.035    0.000    0.944    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/numpy/core/numeric.py:2355(allclose)
	    21513    0.096    0.000    0.812    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/numpy/core/numeric.py:2431(isclose)
	       15    0.149    0.010    0.766    0.051 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:1777(cdf)

	But this means that MPI parallelisation over x_new is no longer possible, so we MPI parallelise over X_new for further linear with processes speedup. Fantastic!



number of gibbs samples = 100
	After taking all monte carlo random variables at once  (Optimisation 1)
	 22279829 function calls (22248734 primitive calls) in 42.549 seconds  
	   Ordered by: cumulative time
	   List reduced from 2077 to 104 due to restriction <0.05>
	   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
	      400    1.633    0.004   34.005    0.085 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:245(predict)
	    40000    4.301    0.000   21.332    0.001 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:157(expectation_wrt_u)
	   160000    2.740    0.000   11.091    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:931(rvs)
	    40000    2.664    0.000    9.537    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:1777(cdf)
		2    0.219    0.109    7.061    3.531 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:113(sample)
	    35569    0.058    0.000    6.461    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_multivariate.py:635(rvs)
	    35569    1.079    0.000    6.182    0.000 {method 'multivariate_normal' of 'mtrand.RandomState' objects}
	   651110    1.272    0.000    5.122    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py:69(_wrapreduction)

	After vectorising over X_new (as opposed to calculating x_new in a loop). (Optimisation 2 = further >2 times speedup)
		 5856278 function calls (5826597 primitive calls) in 15.856 seconds
	   Ordered by: cumulative time
	   List reduced from 2044 to 102 due to restriction <0.05>
	   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
		2    0.236    0.118    7.596    3.798 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:113(sample)
	    38879    0.063    0.000    6.953    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_multivariate.py:635(rvs)
		1    0.185    0.185    6.916    6.916 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:208(predict_vector)
	      100    0.606    0.006    6.694    0.067 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:178(multidimensional_expectation_wrt_u)
	    38879    1.183    0.000    6.643    0.000 {method 'multivariate_normal' of 'mtrand.RandomState' objects}
	      100    1.023    0.010    5.339    0.053 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:1777(cdf)
	    38879    1.606    0.000    3.672    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/linalg/decomp_svd.py:16(svd)
	    38879    0.064    0.000    1.718    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/numpy/core/numeric.py:2355(allclose)
	      100    0.046    0.000    1.689    0.017 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:520(argsreduce)
	      100    0.100    0.001    1.642    0.016 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:545(<listcomp>)
	      100    0.000    0.000    1.595    0.016 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:226(_cdf)
	      100    1.595    0.016    1.595    0.016 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_continuous_distns.py:169(_norm_cdf)

	But this means that MPI parallelisation over x_new is no longer possible, so we MPI parallelise over X_new for further linear with processes speedup. Fantastic!
