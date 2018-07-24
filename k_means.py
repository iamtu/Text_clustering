import sys
import numpy as np
import random
from random import sample

class K_means(object):
    def __init__(self, X, K, centroids_init):
        self.X = X
        self.K = K
        (self.N, self.D) = X.shape
        print 'number of samples: ', self.N, 'dimension: ', self.D
        print 'number of clusters: ', self.K

        self.centroids = centroids_init if centroids_init != None else X[sample(range(self.N), self.K)]
        print 'init: ', self.centroids
    def cluster(self): 
        print 'clustering ...'
        old_centroids = np.zeros((self.centroids.shape))
        iter = 0
        while iter < 1000 or np.linalg.norm(self.centroids - old_centroids) > 1e-6:
            if iter %100 == 0:
                print 'iter ', iter
            old_centroids = self.centroids
            distances = self.find_closest_centroids(self.X, old_centroids)
            self.centroids = self.update_centroids(X, distances, old_centroids)
            iter += 1

    def find_closest_centroids(self, points, centroids):
        """returns an array containing the index to the nearest centroid for each point"""
    #    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    #    return  np.argmin(distances, axis=0)

        closest_indes = np.zeros(points.shape[0])
        for i in xrange(points.shape[0]):
            point = points[i]
            diffs_to_centroids = point - centroids
            distances = np.sqrt(np.sum(diffs_to_centroids**2, axis=-1))
            nearest = np.argmin(distances)
            closest_indes[i] = nearest
        return closest_indes


    def update_centroids(self, points, closest, centroids):
        """returns the new centroids assigned from the points closest to them"""
        return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

if __name__ == '__main__':
    X = np.array([
        [1,1,1],
        [2,2,2],
        [3,3,3],
        [8,8,8],
        [9,9,9]])
    centroids_init = np.array([
        [0,0,0],
        [10,10,10]])
    model = K_means(X, 2, centroids_init = None)
    model.cluster()
    print model.centroids

