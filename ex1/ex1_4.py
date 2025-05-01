import numpy as np
import dataset
import classifier

X4 = dataset.generate_dataset(1000, [np.array([1, 1]), np.array([5, 10]), np.array([1, 11])],
                      [np.array([[5, 4], [4, 7]]), np.array([[5, 4], [4, 7]]), np.array([[5, 4], [4, 7]])],
                      [0.334, 0.333, 0.333])
y = dataset.generate_labelset(1000, np.array([0.334, 0.333, 0.333]))

X4_b = classifier.bayesian(X4,[np.array([1, 1]), np.array([5, 10]), np.array([1, 11])],
                [np.array([[5, 4], [4, 7]]), np.array([[5, 4], [4, 7]]), np.array([[5, 4], [4, 7]])],
                [0.334, 0.333, 0.333])
X4_m = classifier.mahalanobis(X4, [np.array([1, 1]), np.array([5, 10]), np.array([1, 11])],
                [np.array([[5, 4], [4, 7]]), np.array([[5, 4], [4, 7]]), np.array([[5, 4], [4, 7]])])
X4_e = classifier.euclidean(X4, [np.array([1, 1]), np.array([5, 10]), np.array([1, 11])])

X4_b_err = classifier.compute_error(X4_b, y)
X4_m_err = classifier.compute_error(X4_m, y)
X4_e_err = classifier.compute_error(X4_e, y)

print("X4_b_err: ", X4_b_err)
print("X4_m_err:", X4_m_err)
print("X4_e_err:", X4_e_err)