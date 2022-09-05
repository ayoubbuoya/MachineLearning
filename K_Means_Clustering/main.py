from numpy import array
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale  # to normalize data
from sklearn.datasets import load_digits


digits = load_digits()
data = scale(digits.data)

# clusters  because we have 10 digits, init is a system that determine the starting points  of centroids
model = KMeans(n_clusters=10, init="random")
model.fit(data)

print(scale(model.predict(data)))

print(scale(digits.target))
