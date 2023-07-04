import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

class KM():
    def __init__(self):
        self.n_samples = 1500
        self.random_state = 170
        self.X, self.y = make_blobs(n_samples=self.n_samples, random_state=self.random_state)
        self.X = np.vstack((self.X[self.y == 0][:500], self.X[self.y == 1][:100], self.X[self.y == 2][:10]))

    def run(self):
        y_pred = KMeans(n_clusters=3, random_state=self.random_state).fit_predict(self.X)
        return self.X ,y_pred

if __name__ == '__main__':
    km = KM()
    X, y_pred = km.run()
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()
#
#
#
#
#
#
# #
# plt.figure(figsize=(5, 5))
# plt.subplot(221)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.title("Incorrect Number of Blobs")
#
# # Anisotropicly distributed data
# transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
# X_aniso = np.dot(X, transformation)
# y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)
#
# plt.subplot(222)
# plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
# plt.title("Anisotropicly Distributed Blobs")
#
# # Different variance
# X_varied, y_varied = make_blobs(
#     n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
# )
# y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)
#
# plt.subplot(223)
# plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
# plt.title("Unequal Variance")
#
# # Unevenly sized blobs
# X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
# y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_filtered)
#
# plt.subplot(224)
# plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
# plt.title("Unevenly Sized Blobs")
#
# plt.show()
