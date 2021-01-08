number of gibbs samples = 15 
	Before taking all monte carlo random variables at once

		201101342 function calls (201070089 primitive calls) in 313.126 seconds
	   Ordered by: cumulative time
	   List reduced from 2084 to 104 due to restriction <0.05>
	   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
	      400    1.199    0.003  307.396    0.768 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:270(predict)
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

	There is probably more stuff I can do by vectorising over X_new (as opposed to calculating x_new in a loop).
	But this means that MPI parallelisation over x_new is no longer possible. Will leave as is for now.


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

	After doing a vectorisation over X_new (Optimisation 2 = further 2 times speedup)
	  10890191 function calls (10860316 primitive calls) in 22.907 seconds
	   Ordered by: cumulative time
	   List reduced from 2043 to 102 due to restriction <0.05>
	   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
		1    0.588    0.588   13.307   13.307 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:208(predict_vector)
		2    0.249    0.125    8.151    4.075 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:113(sample)
	    37669    0.066    0.000    7.464    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_multivariate.py:635(rvs)
	      100    0.638    0.006    7.404    0.074 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:178(multidimensional_expectation_wrt_u)
	    37669    1.277    0.000    7.137    0.000 {method 'multivariate_normal' of 'mtrand.RandomState' objects}
	      100    1.139    0.011    5.918    0.059 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:1777(cdf)
	   120100    1.412    0.000    5.083    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_distn_infrastructure.py:931(rvs)
	    37669    1.771    0.000    3.974    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/linalg/decomp_svd.py:16(svd)
