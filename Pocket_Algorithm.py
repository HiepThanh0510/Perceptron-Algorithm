import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


def Convert(x):
  if x >= 0: return 1
  else: return 0

def Convergence(x):
    for i in range(len(x)): 
        x[i] = Convert(x[i])
    return x

def Count(Dot_Product, y):
    count = 0
    for m, n in zip(Dot_Product, y):
        if m != n:
            count += 1
    return count

def Perceptron_Algorithm(X, y, loop):
    #P <-- inputs with label 1 
    #N <-- inputs with label 0 
    P = X[:, :][y == 1]
    N = X[:, :][y == 0]
    #Initialize w randomly
    w = np.random.randn(3, )
    Matrix_One = np.ones((X.shape[0], 1))
    X_temp = np.concatenate((Matrix_One, X), axis = 1)
    #Put w in pocket 
    pocket = [w.copy()]
    misclassified = [X.shape[0]]
    #Loop 150 times 
    for loop in range(loop):
        for i in range(X.shape[0]):
            #Pick random x E (P U N)
            X_random = X[i]
            if (X_random in P) and (np.dot(X_temp[i], w)) < 0:
                w = w + X_temp[i]
            if (X_random in N) and (np.dot(X_temp[i], w)) >= 0:
                w = w - X_temp[i]
            #Append old pocket to new w 
            pocket.append(w.copy())
            Dot_Product = Convergence(np.dot(X_temp, w))
            misclassified.append(Count(Dot_Product, y))
    #Finding smallest value in misclassified array 
    index_min = misclassified.index(min(misclassified))
    return pocket[index_min], misclassified[index_min]


if __name__ == "__main__":
    X, Y = datasets.make_blobs(n_samples = 500, n_features = 2, centers = 2,
                               cluster_std = 2.2, random_state = 2)
    Size_Loop = 150
    w, mis_classifies = Perceptron_Algorithm(X, Y, Size_Loop)
    plt.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], 'r^')
    plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], 'bs')
    plt.xlabel("x1")
    plt.ylabel('x2')
    plt.title(f"Pocket Algorithm\nw = {w}\nMiss classifies point: {format(mis_classifies)}/{format(X.shape[0])}"
              + "\n" + f"Accuracy of Pocket Algorithm is {(X.shape[0] - mis_classifies)*100/X.shape[0]}" + "%")
    x1 = np.array([min(X[:, 0]), max(X[:, 0])])
    #w0 + w1x1 + w2x2 = 0
    x2 = -w[1] * x1 / w[2] - w[0] / w[2]
    plt.plot(x1, x2, 'k') #Draw staight line  
    plt.savefig('Pocket_Algorithm.png')
    plt.show()


