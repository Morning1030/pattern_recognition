import numpy as np
# create dataset
np.random.seed(42)
def generate_dataset(means, covariance, N):
    dataset = np.random.multivariate_normal(means, covariance, N)
    # print(dataset.shape)
    return dataset
class_1_a = generate_dataset(np.array([1, 1]), np.array([[1, 0.4], [0.4, 1]]), 150)
class_2_a = generate_dataset(np.array([5, 5]), np.array([[1, -0.6], [-0.6, 1]]), 300)
class_3_a = generate_dataset(np.array([9, 1]), np.eye(2), 100)

X = np.vstack((class_1_a, class_2_a, class_3_a))
class_1_b = generate_dataset(np.array([1, 1]), np.array([[1, 0.4], [0.4, 1]]), 150)
class_2_b = generate_dataset(np.array([2, 2]), np.array([[1, -0.6], [-0.6, 1]]), 300)
class_3_b = generate_dataset(np.array([3, 1]), np.eye(2), 100)

X_b = np.vstack((class_1_b, class_2_b, class_3_b))
y = np.concatenate((np.repeat(1, 150), np.repeat(2, 300), np.repeat(3, 100)))