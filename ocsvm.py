from sklearn.svm import OneClassSVM
import IPython
import numpy as np

class OCSVM():

    def __init__(self, kernel='rbf', nu = .01, gamma = .01):
        self.svm = OneClassSVM(nu = nu, gamma = gamma, kernel=kernel)

    def predict(self, X):
        X_prime = self.mapping(X)
        return self.svm.predict(X_prime)

    def fit(self, X):
        X_prime = self.mapping(X)
        return self.svm.fit(X_prime)

    def decision_function(self, X):
        X_prime = self.mapping(X)
        return self.svm.decision_function(X_prime)

    def mapping(self, X):
        X = np.array(X)
        clutter1_com = X[:, -3:]
        clutter2_com = X[:, -6:-3]
        obj_com = X[:, -12:-9]
        gripper_com = X[:, -15:-12]

        diff1 = clutter1_com - obj_com
        diff2 = clutter2_com - obj_com

        norm1 = np.linalg.norm(diff1, axis=1)
        norm2 = np.linalg.norm(diff2, axis=1)

        X_prime = np.array([norm1, norm2]).T
        X_prime = np.hstack((X_prime, gripper_com))
        return X_prime