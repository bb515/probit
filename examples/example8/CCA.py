"""CCA."""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

iris = datasets.load_iris()

X = iris.data
y = iris.target
print(np.shape(X))
print(np.shape(y))

target_names = iris.target_names

n_samples = len(y)

cca = CCA(n_components=2)
# Centre data
cca.fit(X, Y)


X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
cca = CCA(n_components=2)
cca.fit(X, Y)
# CCA(n_components=1)
X_c, Y_c = cca.transform(X, Y)

print(np.mean(X_c, axis=0))
print(np.mean(Y_c, axis=0))

print(np.mean(X, axis=0))
print(np.mean(Y, axis=0))


cov_matrix = np.dot(X_c.T, X_c) / n_samples
for eigenvector in cca.components_:
    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))


assert 0

pca = PCA(n_components=4)
X_r = pca.fit(X).transform(X)

# Center the data and compute the sample covariance matrix
X -= np.mean(X, axis=0)
cov_matrix = np.dot(X.T, X) / n_samples
for eigenvector in pca.components_:
    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))


lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()

