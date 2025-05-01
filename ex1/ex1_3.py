import numpy as np
import dataset
import classifier

X3 = dataset.generate_dataset(1000, [np.array([1, 1]), np.array([6, 8]), np.array([1, 13])],
                      [6 * np.eye(2), 6 * np.eye(2), 6 * np.eye(2)],
                      [0.334, 0.333, 0.333])
y = dataset.generate_labelset(1000, np.array([0.334, 0.333, 0.333]))

X3_b = classifier.bayesian(X3, [np.array([1, 1]), np.array([6, 8]), np.array([1, 13])],
                [6 * np.eye(2), 6 * np.eye(2), 6 * np.eye(2)],
                [0.334, 0.333, 0.333])
X3_m = classifier.mahalanobis(X3, [np.array([1, 1]), np.array([6, 8]), np.array([1, 13])],
                [6 * np.eye(2), 6 * np.eye(2), 6 * np.eye(2)])
X3_e = classifier.euclidean(X3, [np.array([1, 1]), np.array([6, 8]), np.array([1, 13])])

X3_b_err = classifier.compute_error(X3_b, y)
X3_m_err = classifier.compute_error(X3_m, y)
X3_e_err = classifier.compute_error(X3_e, y)

print("X3_b_err: ", X3_b_err)
print("X3_m_err:", X3_m_err)
print("X3_e_err:", X3_e_err)