from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def Convert(x):
    if x >= 0: return 1
    return 0

def Convergence(a, b):
    for m, n in zip(a, b):
        if m != n:
            return False
    return True

def Perceptron_Algorithm(X, Y):
    #P <-- inputs with label 1 
    #N <-- inputs with label 0 
    P = X[:, :][Y == 1] 
    N = X[:, :][Y == 0]  

    #Initialize w randomly 
    w = np.random.rand(3, )
    size = len(X)
    while True:
        #axis: 0, 1 and none 
        Matrix_One = np.ones((X.shape[0], 1)) 
        X_temp = np.concatenate((Matrix_One, X), axis = 1)
        #Pick random x E P U N
        i = np.random.randint(0, X.shape[0] - 1)
        X_random = X[i]
        
        if (X_random in P) and (np.dot(X_temp[i], w) < 0): 
            w = w + X_temp[i]
        if (X_random in N) and (np.dot(X_temp[i], w) >= 0):
            w = w - X_temp[i]

        #Check convergence 
        Dot_Product = np.dot(X_temp, w)
        for i in range(size):
            Dot_Product[i] = Convert(Dot_Product[i])
        if Convergence(Dot_Product, Y):
            return w


if __name__ == "__main__":
    #Linearly separable data 
    X, Y = datasets.make_blobs(n_samples = 60, n_features = 2, centers = 2,
                               cluster_std = 2, random_state = 2)
    w = Perceptron_Algorithm(X, Y)  
    print('w = ', w)
    x1 = np.array([min(X[:, 0]), max(X[:, 0])])
    #w0 + w1x1 + w2x2 = 0 
    x2 = -(w[0] + w[1] * x1 ) / w[2]  
    plt.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], 'bs')
    plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], 'r^')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Dataset: N = 60, 0: 0.5, 1: 0.5" + '\n' + f"w = {w}"+ '\n' + "Perceptron 2D")
    plt.plot(x1, x2, 'k') #Draw straight line 
    plt.savefig('2D_perceptron.png', bbox_inches = 'tight')
    plt.show()
