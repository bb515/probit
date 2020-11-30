"""Multinomial Probit regression using Gibbs sampling with GP priors."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, multivariate_normal

def log_heaviside_probit_likelihood(u, t, G):
    """The log(p(t|u)) when t=1 indicates inclass and t=0 indicates outclass."""
    v_sample = np.dot(G, u)
    ones = np.ones(len(v_sample))
    phi = norm.cdf(v_sample)
    one_minus_phi = np.subtract(ones, phi)
    log_one_minus_phi = np.log(one_minus_phi)
    log_phi = np.log(phi)
    log_likelihood = (np.dot(t, log_phi)
                      + np.dot(np.subtract(ones, t), log_one_minus_phi))
    return log_likelihood


def gibbs(M_0, N, X, t):
    """
    Sampling occurs in blocks over the parameters: Y (auxilliaries) and M.
    :param initial: The initial location of the sampler.
    :param N: the number of iterations.
    :param X: The data objects in R^D
    :param t: The target variables
    :return: Gibbs samples and acceptance rate.
    """
    samples = []
    M = M_0
    # Transpose
    X_T = X.T
    cov = np.linalg.inv(X_T @ X)
    for _ in range(N):
        # Empty Y vector to collect Y_i samples over
        Y = []
        for i, x in enumerate(X):
            # Sample from truncated Guassian depending on t
            if t[i] == 1:
                yi = 0
                while yi <= 0:
                    yi = norm.rvs(loc=np.dot(M, x), scale=1)
            else:
                yi = 0
                while yi >= 0:
                    yi = norm.rvs(loc=np.dot(M, x), scale=1)
            # Add sample to the Y vector
            Y.append(yi)

        # Calculate statistics, then sample other conditional
        mean = cov @ X_T @ np.array(Y)
        M = multivariate_normal.rvs(mean=mean, cov=cov)
        samples.append((M, Y))
    return samples

def predict_gibbs(beta_samples, x):
    x = np.array([1,x[0],x[1]])
    f = [norm.cdf(np.dot(beta,x)) for beta in beta_samples]
    return sum(f)/len(beta_samples)

# Rough number of datapoints to generate for each class
n = 50

# Uniform quadrant dataset - linearly seperable
X0 = np.random.rand(50, 2)
X1 = np.ones((50, 2)) + np.random.rand(50, 2)
offset = np.array([0, 1])
offsets = np.tile(offset, (50, 1))
X2 = offsets + np.random.rand(50, 2)
#X0 = np.array([(np.random.rand(),np.random.rand()) for _ in range(n)])
#X1 = np.array([(1+np.random.rand(),1+np.random.rand()) for _ in range(n)])
t0 = np.zeros(len(X0))
t1 = np.ones(len(X1))
t2 = np.ones(len(X2))

## Plotting
plt.scatter(X0[:,0], X0[:,1], color='b', label=r"$t=0$")
plt.scatter(X1[:,0], X1[:,1], color='r', label=r"$t=1$")
plt.scatter(X2[:,0], X2[:,1], color='g', label=r"$t=2$")
plt.legend()
plt.xlim(0,2)
plt.ylim(0,2)
plt.xlabel(r"$x_1$",fontsize=16)
plt.ylabel(r"$x_2$",fontsize=16)
plt.show()

# Extend the input vectors
X0_tilde = np.c_[np.ones(len(X0)),X0]
X1_tilde = np.c_[np.ones(len(X1)),X1]

# prepare data
X = np.r_[X0_tilde, X1_tilde]
t = np.r_[t0, t1]

# Shuffle data - may only be necessary for matrix conditioning
Xt = np.c_[X,t]
np.random.shuffle(Xt)
X = Xt[:,:3]
t = Xt[:,-1]

# Sample beta from prior
beta = multivariate_normal.rvs(mean=[0,0,0], cov=np.eye(3))

# Take n samples, returning the beta and y samples
n = 500
samples = gibbs(beta, n, X, t)

beta_samples = np.array([sample[0] for sample in samples])
norm_samples = np.array([beta/np.linalg.norm(beta) for beta in beta_samples])
y_samples = np.array([sample[1] for sample in samples])

fig, ax = plt.subplots(1,3, figsize=(15,5))
beta_star = np.zeros(3)
n0, b0, patches = ax[0].hist(norm_samples[:,0], 20, density="probability", histtype='stepfilled')
n1, b1, patches = ax[1].hist(norm_samples[:,1], 20, density="probability", histtype='stepfilled')
n2, b2, patches = ax[2].hist(norm_samples[:,2], 20, density="probability", histtype='stepfilled')
beta_star[0] = b0[np.where(n0 == n0.max())]
beta_star[1] = b1[np.where(n1 == n1.max())]
beta_star[2] = b2[np.where(n2 == n2.max())]
ax[0].axvline(beta_star[0],color='k',label=r"Maximum $\beta$")
ax[1].axvline(beta_star[1],color='k',label=r"Maximum $\beta$")
ax[2].axvline(beta_star[2],color='k',label=r"Maximum $\beta$")
ax[0].set_xlabel(r"$\beta_0$",fontsize=16)
ax[1].set_xlabel(r"$\beta_1$",fontsize=16)
ax[2].set_xlabel(r"$\beta_2$",fontsize=16)
ax[0].legend()
ax[1].legend()
ax[2].legend()
plt.show()

N = 20
x, y = np.mgrid[-0.1:2.1:2.2/N, -0.1:2.1:2.2/N]
pos = np.dstack((x, y))

Z = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        Z[i,j] = predict_gibbs(beta_samples, np.array( (x[i,j], y[i,j]) ) )

fig, axs = plt.subplots(1,figsize=(6,6))
plt.scatter(X0[:,0], X0[:,1], color='b', label=r"$t=0$", zorder=10)
plt.scatter(X1[:,0], X1[:,1], color='r', label=r"$t=1$", zorder=10)
plt.contourf(x,y,Z,zorder=1)
plt.xlim(0,2)
plt.ylim(0,2)
plt.legend()
plt.xlabel(r"$x_1$",fontsize=16)
plt.ylabel(r"$x_2$",fontsize=16)
plt.title("Contour plot - Gibbs")
plt.show()
