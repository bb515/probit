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


After taking all monte carlo random variables at once
         6504295 function calls (6473070 primitive calls) in 10.013 seconds
   Ordered by: cumulative time
   List reduced from 2084 to 104 due to restriction <0.05>
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      400    0.492    0.001    4.433    0.011 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:273(predict)
        2    0.122    0.061    4.075    2.037 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:114(sample)
    21343    0.034    0.000    3.734    0.000 /home/ben/anaconda3/lib/python3.7/site-packages/scipy/stats/_multivariate.py:635(rvs)
    21343    0.629    0.000    3.567    0.000 {method 'multivariate_normal' of 'mtrand.RandomState' objects}
     6000    0.505    0.000    2.513    0.000 /home/ben/multivariate_regression_GP_gibbs/probit/samplers.py:167(expectation_wrt_u)

There is probably more stuff I can do by vectorising over X_new (as opposed to calculating x_new in a loop). But this means that MPI parallelisation over x_new is no longer possible. Will leave as is for now.:S


