from sklearn import datasets
import KMedoid as kmd
import unittest
import random
import matplotlib.pyplot as plt

class kMedoidTest(unittest.TestCase):
    def setUp(self):
        random.seed(35)
        self.iris = datasets.load_iris()
        self.X = self.iris.data[:, :2]

    def whenInitializingKwith2MustReturn2MedoidsWithinXRange(self):
        medoids = kmd.initializeKMedoids(2, self.X)
        self.assertEqual(len(medoids), 2)
        for medoid in medoids.values():
            self.assertIn(medoid, self.X)

    def WhenAssigningKMedoidsClustersMustBeInsideKRange(self):
        K = 2
        medoids = kmd.initializeKMedoids(K, self.X)
        clusters = kmd.assignKMedoids(K, self.X, medoids)
        for k in range(0,K):
            self.assertIn(k, clusters.values())

    def WhenUpdatingKMedoidsMustReturnMedoidsWithinXRange(self):
        K = 2
        medoids = kmd.initializeKMedoids(K, self.X)
        clusters = kmd.assignKMedoids(K, self.X, medoids)
        updatedMedoids = kmd.updateKMedoids(K, self.X, clusters)
        for medoid in updatedMedoids.values():
            self.assertIn(medoid, self.X)


    def WhenComparingTwoEqualMedoidsAreEqualShouldReturnTrue(self):
        K = 2
        medoids = kmd.initializeKMedoids(K, self.X)
        self.assertTrue(kmd.medoidsAreEqual(K, medoids, medoids))

    def ShowPlot(self):
        K = 3
        kmd.plotKMedoid(3, self.X)
        plt.show()