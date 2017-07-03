import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import mlab

def initializeKMedoids(K, X):
    medoids = {}
    points = X[random.sample(range(0, len(X)), K),:]
    for k in range(0, K):
        medoids[k] = points[k]
    return medoids


def assignKMedoids(K, X, medoids):
    clusterX = {}
    for i in range(0, len(X)):
        minDistance = sys.maxsize
        x = X[i]
        for k in range(0, K):
            medoid = medoids[k]
            distance = getDistanceBetweenPoints(medoid, x)
            if(distance < minDistance):
                minDistance = distance
                clusterX[i] = k
    return clusterX


def getDistanceBetweenPoints(point1, point2):
    return np.linalg.norm(point1 - point2)


def getAverageDistanceFromMedoid(medoid, points):
    distances = []
    for point in points:
        distances.append(getDistanceBetweenPoints(medoid, point))
    return np.mean(distances)


def updateKMedoids(K, X, clustersX):
    newMedoids = {}
    for k in range(0, K):
        medoid = {"distance": sys.maxsize, "point": []}
        xIndexes = [i for key, i in clustersX.items() if key == k]
        for idx in xIndexes:
            distance = getAverageDistanceFromMedoid(X[idx], X[xIndexes])
            if distance < medoid["distance"]:
                medoid["distance"] = distance
                medoid["point"] = X[idx]
        newMedoids[k] = medoid["point"]
    return newMedoids


def trainKMedoid(K, X, t = 30):
    medoids = initializeKMedoids(K, X)
    for _ in range(0, t):
        clustersX = assignKMedoids(K, X, medoids)
        newMedoids = updateKMedoids(K, X, clustersX)
        if medoidsAreEqual(K, medoids, newMedoids):
            break
        medoids = newMedoids
    return medoids


def medoidsAreEqual(K, oldMedoids, newMedoids):
    for k in range(1,K):
        oldMedoid = oldMedoids[k]
        newMedoid = newMedoids[k]
        for i in range(0,len(oldMedoid)):
            old = oldMedoid[i]
            new = newMedoid[i]
            if old != new:
                return False
    return True


def plotKMedoid(K, X):
    # Used demo from https://stackoverflow.com/questions/9847026/plotting-output-of-kmeanspycluster-impl
    # cluster
    kMedoids = trainKMedoid(K, X)
    clustersIds = assignKMedoids(K, X, kMedoids)

    # reduce dimensionality
    iris_pca = mlab.PCA(X)
    cutoff = iris_pca.fracs[1]
    iris_2d = iris_pca.project(X, minfrac=cutoff)
    medoids_2d = iris_pca.project(list(kMedoids.values()), minfrac=cutoff)

    # make a plot
    colors = ['red', 'green', 'blue', 'yellow']
    plt.figure()
    plt.xlim([iris_2d[:, 0].min() - .5, iris_2d[:, 0].max() + .5])
    plt.ylim([iris_2d[:, 1].min() - .5, iris_2d[:, 1].max() + .5])
    plt.xticks([], [])
    plt.yticks([], [])  # numbers aren't meaningful

    # show the centroids
    plt.scatter(medoids_2d[:, 0], medoids_2d[:, 1], marker='o', c=colors, s=100)

    # show user numbers, colored by their cluster id
    for i, ((x, y), kls) in enumerate(zip(iris_2d, list(clustersIds.values()))):
        plt.annotate(str(i), xy=(x, y), xytext=(0, 0), textcoords='offset points',
                     color=colors[kls])