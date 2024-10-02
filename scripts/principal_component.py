import numpy as np
import matplotlib.pyplot  as plt
from sklearn.decomposition import PCA

def principal_component(X_train):
    # Perform PCA on the training data 
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    # Print the explained varaince ration for each principal component 
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Plot the training data in the reduced dimensionality space
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
    plt.title("PCA on Training Data")
    plt.show()
    
    return X_train_pca
