import numpy as np

from pandas import read_csv, DataFrame, concat
import matplotlib.pyplot as plt
from gower import gower_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors as knn
from sklearn.metrics import silhouette_score, rand_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import squareform as sf
import seaborn as sns
from kneed import KneeLocator


def tSNE_plot(tsne_results, LABELS, legend=["Outliers"], title="", return_plot= False):
    """
    INPUTS:

        tsne_results - Some sklearn TSNE result already computed

        LABELS - Used to set markers and colors in plot, no predefined range, the first label is represented as black

        legend - Labels to set in legend, the default considers the first (in numerical order) label as associated to outliers

    """
    # visualization of the datased provided a tsne_result
    fig = plt.figure(figsize=(10, 10))
    plt.title(title)
    PAL = [(0, 0, 0)] + sns.color_palette(None, n_colors=(np.unique(LABELS).shape[0]-1))  # set palette, first color: black
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=LABELS, style=LABELS,
                    palette=PAL, ax=fig.gca())  # scatterplot, random markers and colors
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(legend)

    plt.show()

    if return_plot:
        return fig


def intra_cluster_distances(labels, DM):
    """
    INPUTS:

        labels - Clustering solution, ranging from 0 to K-1

        DM - Pairwise distance matrix to calculate the intra cluster distance with
    """
    K = max(labels)+1 # number of clusters
    intra_dist=[] # initialize
    for k in range(K): # for each cluster
        pairwise_dist=[] # pairwise distance as list of all pair distances intra-cluster
        cluster = [i for i, x in enumerate(labels==k) if x] # indexes of points associated to the cluster
        remaining = cluster # the points we want to measure the pair distance
        for p in cluster: # for each point in the cluster
            remaining = [q for q in remaining if q is not p] # remove p from the cluster
            if len(remaining) == 0: # if no points remained: break
                break
            pairwise_dist += DM[p, remaining].tolist() # concatenate the pairwise distances that we did not calculate yet

        intra_dist.append(np.mean(pairwise_dist)) # take the mean of all intra-cluster pairwise distances and append to the list

    return intra_dist

def inter_cluster_distances(labels, DM, method="average"):
    """
    INPUTS:

        labels - Clustering solution, ranging from 0 to K-1

        DM - Pairwise distance matrix to calculate the intra cluster distance with

        method - Cluster distance criterion
    """
    K = max(labels)+1 # number of cluster
    inter_dist=[] # the pairwise distances between clusters
    for k1 in range(K): # select first cluster
        cluster1 = [i for i, x in enumerate(labels==k1) if x] # get indexes of first cluster

        for k2 in range(k1+1, K): # select second cluster, we compare just from k1+1
            cluster2 = [i for i, x in enumerate(labels==k2) if x] # get indexes of second cluster
            pairwise_distances=[] # reset pairwise distance of points in the two clusters
            for x in cluster1:
                for y in cluster2:
                    pairwise_distances.append(DM[x,y].item()) # take all pairwise distances of points in the two clusters

            # perform the cluster distance
            if method == "maximum":
                inter_dist.append(max(pairwise_distances))
            if method == "minimum":
                inter_dist.append(min(pairwise_distances))
            if method == "average":
                inter_dist.append(np.mean(pairwise_distances))

    # since we appended for every cluster the distance
    # to the subsequents we have a condensed-form distance matrix
    # finally turn it into square form
    inter_dist = sf(inter_dist)
    return inter_dist


#################################################################################
#                         PLOTS UTILS                                           #
#################################################################################

def histogram(X, bins=None, density=False, title="", labelx="", labely="", grid=False, xticks=None,
              yticks=None, alpha=0.5, label=None):
    """
    Rapid function to plot histograms.

    plt.figure and plt.show are left outside this function in order to further manipulate the plot (e.g. with subplots).
    """

    if bins is None:
        bins = 10


    plt.title(title)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if grid:
        plt.grid(alpha= alpha)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.hist(X, bins=bins, density=density)



def imshow(X, title=""):
    """
    Rapid function for imshow.

    plt.figure and plt.show are left outside this function in order to further manipulate the plot (e.g. with subplots).
    """
    plt.title(title)
    plt.imshow(X)
    plt.colorbar()


def barplot(X, Y, title="", labelx="", labely="", grid=True, xticks=None, yticks=None):
    """
    Rapid function for barplots.

    plt.figure and plt.show are left outside this function in order to further manipulate the plot (e.g. with subplots).
    """
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.grid(visible=grid)
    plt.bar(X, Y)


def plot(x, y=None, title="", labelx="", labely="", xlim=None, ylim=None, grid=True, xticks=None, yticks=None, label=None):
    """
    Rapid function for line plots.

    plt.figure and plt.show are left outside this function in order to further manipulate the plot (e.g. with subplots).
    """
    if y is None:
        y = x
        x = range(len(y))

    plt.plot(x, y, label=label)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.grid(visible=grid)

