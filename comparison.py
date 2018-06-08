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

# Reduce dimension to 2 with NeighborhoodComponentAnalysis
nca_sklearn = Pipeline([('scaler', StandardScaler()),
                        ('nca', NeighborhoodComponentsAnalysis(
                                                   n_features_out=2,
                                                   random_state=random_state,
                                                   store_opt_result=True))])
pca = PCA()

fast_nca =  make_pipeline(StandardScaler(), NCA(dim=2))  #note that NCA here
#  is modified to use PCA as initialisation

# Use a nearest neighbor classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# Make a list of the methods to be compared
dim_reduction_methods = [('pca', pca), ('fast_nca', fast_nca),
                                                  ('nca_sklearn',
                                                      nca_sklearn)]

# plt.figure()
for i, (name, model) in enumerate(dim_reduction_methods):
    plt.figure()
    # plt.subplot(1, 3, i + 1, aspect=1)

    # Fit the method's model
    model.fit(X_train, y_train)

    # Fit a nearest neighbor classifier on the embedded training set
    knn.fit(model.transform(X_train), y_train)

    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(model.transform(X_test), y_test)

    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = model.transform(X)

    # Plot the embedding and show the evaluation score
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name,
                                                              n_neighbors,
                                                              acc_knn))
plt.show()

print(nca_sklearn.named_steps['nca'].opt_result_)