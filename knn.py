import numpy as np

class KNN():

    def __init__(self, k):
        self.k = k

    def euclidean(self, X, y):
        distance = 0.0
        for i in range(len(X)-1):
            distance += (X[i] - y[i]) ** 2

        return distance ** 1/2

    def fit(self, train, test):
        distances = []
        for row in train:
            distance = euclidean(test, train)
            distances.append((row, distance))

        distances.sort(key=lambda x: x[1])
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])

        return neighbors


    def predict(train, test):
        neighbors = fit(train, test, self.k)
        out = [row[-1] for row in neighbors]
        pred = max(set(out), key=out.count)

        return pred
