import numpy as np

# generate dataset
# N = 1000, two classes, 3D vector
def generate_dataset(N, mean, cov):
    print(mean)
    print(cov)
    x = np.random.multivariate_normal(mean, cov, N)
    # print(x)
    return x

def ml_estimator_individual(dataset):
    mean_indi = np.mean(dataset, axis=0)
    var_indi = np.var(dataset, axis=0)
    return mean_indi, var_indi

#unknown mean and covariance
def ml_estimator_gaussian(dataset):
    N = dataset.shape[0]
    mean_gau = np.mean(dataset, axis=0)
    cov_gau = ((dataset - mean_gau).T @ (dataset - mean_gau)) / N
    return mean_gau, cov_gau
# main
C1 = generate_dataset(1000, np.array([0, 0, 0]), np.diag([3, 5, 2]))
mean_indi, var_indi = ml_estimator_individual(C1)
mean_gau, cov_gau = ml_estimator_gaussian(C1)
print("(a):")
print(mean_indi)
print(var_indi)
print("(b):")
print(mean_gau)
print(cov_gau)


C2 = generate_dataset(1000, np.array([1, 5, -3]), np.array([[1, 0, 0],[0, 4, 1], [0, 1, 6]]))
mean_indi, var_indi = ml_estimator_individual(C2)
mean_gau, cov_gau = ml_estimator_gaussian(C2)
print("(d):")
print(mean_indi)
print(var_indi)
print("(e):")
print(mean_gau)
print(cov_gau)
