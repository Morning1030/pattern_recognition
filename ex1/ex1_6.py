import numpy as np
import dataset
import classifier

X6 = dataset.generate_dataset(1000, [np.array([1, 1]), np.array([7, 7]), np.array([1, 15])],
                      [np.array([[1, 0], [0, 12]]), np.array([[2, 3], [3, 8]]), np.array([[2, 0], [0, 2]])],
                      [0.334, 0.333, 0.333])



X6_prime = dataset.generate_dataset(1000, [np.array([1, 1]), np.array([7, 7]), np.array([1, 15])],
                      [np.array([[1, 0], [0, 12]]), np.array([[2, 3], [3, 8]]), np.array([[2, 0], [0, 2]])],
                      [0.6, 0.3, 0.1])

y = dataset.generate_labelset(1000, np.array([0.334, 0.333, 0.333]))
y6_prime = dataset.generate_labelset(1000, np.array([0.6, 0.3, 0.1]))

X6_b = classifier.bayesian(X6, [np.array([1, 1]), np.array([7, 7]), np.array([1, 15])],
                [np.array([[1, 0], [0, 12]]), np.array([[2, 3], [3, 8]]), np.array([[2, 0], [0, 2]])],
                [0.334, 0.333, 0.333])
X6_e = classifier.euclidean(X6, [np.array([1, 1]), np.array([7, 7]), np.array([1, 15])])

X6_prime_b = classifier.bayesian(X6_prime, [np.array([1, 1]), np.array([7, 7]), np.array([1, 15])],
                [np.array([[1, 0], [0, 12]]), np.array([[2, 3], [3, 8]]), np.array([[2, 0], [0, 2]])],
                [0.6, 0.3, 0.1])
X6_prime_e = classifier.euclidean(X6_prime, [np.array([1, 1]), np.array([7, 7]), np.array([1, 15])])
# print(X1)
# print(y)
# print(X1_b)

X6_b_err = classifier.compute_error(X6_b, y)
X6_e_err = classifier.compute_error(X6_e, y)
X6_prime_b_err = classifier.compute_error(X6_prime_b, y6_prime)
X6_prime_e_err = classifier.compute_error(X6_prime_e, y6_prime)

print("X6_b_err: ", X6_b_err)
print("X6_e_err:", X6_e_err)
print("X6_prime_b_err: ", X6_prime_b_err)
print("X6_prime_e_err:", X6_prime_e_err)