from metric_learn import RCA_Supervised, RCA
from nca import NCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, \
    NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from nca import NCA

print(__doc__)

n_neighbors = 1
random_state = 0

# Load Digits dataset
digits = datasets.fetch_olivetti_faces()
X, y = digits.data, digits.target

# Split into train/test
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, stratify=y,
                     random_state=random_state)

dim = len(X[0])
n_classes = len(np.unique(y))

knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# intialize algorithm
rca = RCA(num_dims=2).fit(X_train, y_train)
model = NeighborhoodComponentsAnalysis(n_features_out=2,
                                       init = rca.transformer(),
                                       verbose = 1,
                                       tol = 1e-5,
                                       max_iter=50,
                                       store_opt_result=True)
X_embedded_train = model.fit_transform(X_train, y_train)
matrix = model.transformation_

print(model.opt_result_)

knn.fit(X_embedded_train, y_train)

acc_knn = knn.score(model.transform(X_test), y_test)

X_embedded = model.transform(X)

# Plot the embedding and show the evaluation score
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format('nca',
                                                          n_neighbors,
                                                          acc_knn))

plt.show()
