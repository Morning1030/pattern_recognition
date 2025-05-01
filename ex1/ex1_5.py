import numpy as np
import dataset
import classifier

X5 = dataset.generate_dataset(1000, [np.array([1, 1]), np.array([4, 4]), np.array([1, 8])],
                      [2 * np.eye(2), 2 * np.eye(2), 2 * np.eye(2)],
                      [0.334, 0.333, 0.333])
X5_prime = dataset.generate_dataset(1000, [np.array([1, 1]), np.array([4, 4]), np.array([1, 8])],
                      [2 * np.eye(2), 2 * np.eye(2), 2 * np.eye(2)],
                      [0.7, 0.1, 0.2])
y = dataset.generate_labelset(1000, np.array([0.334, 0.333, 0.333]))
y5_prime = dataset.generate_labelset(1000, np.array([0.7, 0.1, 0.2]))

X5_b = classifier.bayesian(X5, [np.array([1, 1]), np.array([4, 4]), np.array([1, 8])],
                [2 * np.eye(2), 2 * np.eye(2), 2 * np.eye(2)],
                [0.334, 0.333, 0.333])
X5_e = classifier.euclidean(X5, [np.array([1, 1]), np.array([4, 4]), np.array([1, 8])])

X5_prime_b = classifier.bayesian(X5_prime, [np.array([1, 1]), np.array([4, 4]), np.array([1, 8])],
                [2 * np.eye(2), 2 * np.eye(2), 2 * np.eye(2)],
                [0.7, 0.1, 0.2])
X5_prime_e = classifier.euclidean(X5_prime, [np.array([1, 1]), np.array([4, 4]), np.array([1, 8])])


X5_b_err = classifier.compute_error(X5_b, y)
X5_e_err = classifier.compute_error(X5_e, y)
X5_prime_b_err = classifier.compute_error(X5_prime_b, y5_prime)
X5_prime_e_err = classifier.compute_error(X5_prime_e, y5_prime)

print("X5_b_err: ", X5_b_err)
print("X5_e_err:", X5_e_err)
print("X5_prime_b_err: ", X5_prime_b_err)
print("X5_prime_e_err:", X5_prime_e_err)