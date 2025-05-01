import numpy as np
import dataset
import classifier
# generating the dataset and grond-truth


X1 = dataset.generate_dataset(1000, [np.array([1, 1]), np.array([8, 12]), np.array([1, 16])],
                      [4 * np.eye(2), 4 * np.eye(2), 4 * np.eye(2)],
                      [0.334, 0.333, 0.333])

y = dataset.generate_labelset(1000, np.array([0.334, 0.333, 0.333]))

X1_b = classifier.bayesian(X1, [np.array([1, 1]), np.array([6, 8]), np.array([1, 13])],
                [6 * np.eye(2), 6 * np.eye(2), 6 * np.eye(2)],
                [0.334, 0.333, 0.333])
X1_m = classifier.mahalanobis(X1, [np.array([1, 1]), np.array([6, 8]), np.array([1, 13])],
                [6 * np.eye(2), 6 * np.eye(2), 6 * np.eye(2)])
X1_e = classifier.euclidean(X1, [np.array([1, 1]), np.array([6, 8]), np.array([1, 13])])

X1_b_err = classifier.compute_error(X1_b, y)
X1_m_err = classifier.compute_error(X1_m, y)
X1_e_err = classifier.compute_error(X1_e, y)

print("X1_b_err: ", X1_b_err)
print("X1_m_err:", X1_m_err)
print("X1_e_err:", X1_e_err)





