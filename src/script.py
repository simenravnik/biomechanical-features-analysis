import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data():
    """
    Reading data sets
    :return: a pair of pandas data frames
    """
    data_2c = pd.read_csv('../data/dataset-2C.csv')
    data_3c = pd.read_csv('../data/dataset-3C.csv')

    return data_2c, data_3c


def column_distribution(x):
    """
    Plotting distribution histogram on which is fitted kernel density estimate (KDE)
    """
    sns.distplot(x)
    plt.show()


def pairwise_relationship(X):
    """
    Plotting multiple pairwise bivariate distributions in a dataset
    """
    sns.pairplot(X)
    sns.pairplot(X, hue="class")
    plt.show()


def class_count_plot(data_2c, data_3c):
    """
    Plotting the count of dataset class for both data sets
    """

    # plotting dataset with normal/abnormal class
    sns.countplot(x="class", data=data_2c, palette="Set3")
    plt.show()

    # plotting dataset with hernia/spondylolisthesis/normal class
    sns.countplot(x="class", data=data_3c, palette="Set3")
    plt.show()


def hierarchical_clustering(data_3c):

    # convert pandas data frame to numpy array
    data_matrix = data_3c.to_numpy()

    # remove class column from the data
    data = data_matrix[:, 0:-1]
    class_column = data_3c.iloc[:, -1]

    # plotting dendogram of hierarchical clustering with ward linkage
    shc.dendrogram(shc.linkage(data, method='ward'))
    plt.show()

    # hierarchical clustering for plotting scatter plot
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)
    plt.scatter(data[:, 1], data[:, 4], c=cluster.labels_, cmap='rainbow')
    plt.show()


def calculate_pca(data):

    # separating the features from the target value
    x = data.iloc[:, 0:-1].values
    target = data.iloc[:, -1].values

    # centering the data (standardizing the features)
    x = StandardScaler().fit_transform(x)

    # performing PCA
    pca = PCA(n_components=2)                           # two components
    principal_components = pca.fit_transform(x)         # fitting the data

    # transforming the data matrix into pd data frame
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # concatenating the target values to the principal component data frame
    final_df = pd.concat([principal_df, data['class']], axis=1)

    # plotting the principal components
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    # defining colors for each or the target values
    targets = ['Hernia', 'Spondylolisthesis', 'Normal']
    colors = ['r', 'g', 'b']

    # for each of the target values ilocate indexes and
    # plotting them on scatter plot with the defined color
    for target, color in zip(targets, colors):

        idx_to_keep = final_df['class'] == target

        ax.scatter(final_df.loc[idx_to_keep, 'PC1']
                   , final_df.loc[idx_to_keep, 'PC2']
                   , c=color
                   , s=50)

    ax.legend(targets)
    ax.grid()

    plt.show()

    # displaying cumulative explained variance ratio depending on number of components
    cumulative_explained_variance_ratio(x)


def cumulative_explained_variance_ratio(x):
    pca = PCA().fit(x)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


if __name__ == '__main__':

    # load data
    data_2c, data_3c = load_data()

    # plotting distribution for each column in dataset
    for column in data_2c.drop('class', axis=1):
        column_distribution(data_2c[column])

    # plotting pairwise relationships
    pairwise_relationship(data_3c)

    # plotting class count
    class_count_plot(data_2c, data_3c)

    hierarchical_clustering(data_3c)

    # dimensionality reduction
    calculate_pca(data_3c)
