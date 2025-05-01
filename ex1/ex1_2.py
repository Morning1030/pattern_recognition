import numpy as np
import dataset
import classifier

X2 = dataset.generate_dataset(1000, [np.array([1, 1]), np.array([7, 14]), np.array([1, 16])],
                      [np.array([[4, 3], [3, 5]]), np.array([[4, 3], [3, 5]]), np.array([[4, 3], [3, 5]])],
                      [0.334, 0.333, 0.333])
y = dataset.generate_labelset(1000, np.array([0.334, 0.333, 0.333]))

X2_b = classifier.bayesian(X2, [np.array([1, 1]), np.array([7, 14]), np.array([1, 16])],
                [np.array([[4, 3], [3, 5]]), np.array([[4, 3], [3, 5]]), np.array([[4, 3], [3, 5]])],
                [0.334, 0.333, 0.333])
X2_m = classifier.mahalanobis(X2, [np.array([1, 1]), np.array([7, 14]), np.array([1, 16])],
                [np.array([[4, 3], [3, 5]]), np.array([[4, 3], [3, 5]]), np.array([[4, 3], [3, 5]])])
X2_e = classifier.euclidean(X2, [np.array([1, 1]), np.array([7, 14]), np.array([1, 16])])

X2_b_err = classifier.compute_error(X2_b, y)
X2_m_err = classifier.compute_error(X2_m, y)
X2_e_err = classifier.compute_error(X2_e, y)

print("X2_b_err: ", X2_b_err)
print("X2_m_err:", X2_m_err)
print("X2_e_err:", X2_e_err)