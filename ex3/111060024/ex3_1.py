# import everything
import matplotlib.image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
training_file = 'mnist_train.csv'

def dataset(fileName):
    # read the data from mnist_train.csv and sample 2000 of them
    mnist_train = pd.read_csv(fileName)
    mnist_train = mnist_train[mnist_train['label'] == 3]
    mnist_train = mnist_train.sample(2000)
    mnist_train = mnist_train.drop(columns='label')
    mnist_train = np.array(mnist_train)
    X_mean = np.average(mnist_train, axis=0)
    mnist_train = (mnist_train - X_mean).T
    return mnist_train, X_mean
    # print(mnist_train.shape)
    # print(len(mnist_train))
def compute_reconstruction_error(X, l):
    w_l = eigen_vector_X[:, -l:]
    Z = w_l.T @ X
    error_sum = np.sum(np.square(X - w_l @ Z), axis=0)
    error = np.average(error_sum)
    return error
def PCA_reconstruction(X, X_mean, l):
    # x = X_mean + sum(zi wi)
    w_l = eigen_vector_X[:, -l:]
    Z = X @ w_l
    result = X_mean + Z @ w_l.T
    result = np.reshape(result, (28, 28))
    return result
    
def eigen_visualization(X, eigen_value, l):
    # X = np.reshape(X, (28, 28))
    min_val = np.min(X)
    max_val = np.max(X)
    # visualize the numpy to a colored image
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_yellow', ['yellow', 'white', 'blue'])
    norm = matplotlib.colors.Normalize(min_val, max_val)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_image = scalar_map.to_rgba(X)
    
    fig, ax = plt.subplots()
    ax.imshow(color_image)
    fig.colorbar(scalar_map, ax=ax, label='Value')
    ax.set_title(f'Visualization of w{l}, eigen_value lambada{l} = {eigen_value}')
    ax.axis('off')  # 不顯示軸
    plt.show()
def mean_visualization(X_mean):
    X_mean = np.reshape(X_mean, (28, 28))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_white', ['white', 'blue'])   
    norm = matplotlib.colors.Normalize(0, np.max(X_mean))
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_image = scalar_map.to_rgba(X_mean)
    fig, ax = plt.subplots()
    ax.imshow(color_image)
    # fig.colorbar(scalar_map, ax=ax, label='Value')
    ax.set_title(f'Mean of x')
    ax.axis('off')  # 不顯示軸
    plt.show()
def PCA_visualization(X, l):
    X_PCA_l = PCA_reconstruction(X, X_mean, l)
    X_PCA_1 = PCA_reconstruction(X, X_mean, 1)
    X_PCA_10 = PCA_reconstruction(X, X_mean, 10)
    X_PCA_50 = PCA_reconstruction(X, X_mean, 50)
    X_PCA_250 = PCA_reconstruction(X, X_mean, 250)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_white', ['white', 'blue'])   
    norm = matplotlib.colors.Normalize(0, np.max(X_mean))
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    image_original = scalar_map.to_rgba(np.reshape(X + X_mean, (28, 28)))
    image_l = scalar_map.to_rgba(X_PCA_l)
    image_1 = scalar_map.to_rgba(X_PCA_1)
    image_10 = scalar_map.to_rgba(X_PCA_10)
    image_50 = scalar_map.to_rgba(X_PCA_50)
    image_250 = scalar_map.to_rgba(X_PCA_250)
    fig, ax = plt.subplots(3, 2)
    # ax_to_remove = ax[0, 1]
    # fig.delaxes(ax_to_remove)
    ax[0, 0].imshow(image_original)
    ax[0, 1].imshow(image_l)
    ax[1, 0].imshow(image_1)
    ax[1, 1].imshow(image_10)
    ax[2, 0].imshow(image_50)
    ax[2, 1].imshow(image_250)
    # fig.colorbar(scalar_map, ax=ax, label='Value')
    ax[0, 0].set_title(f'original image')
    ax[0, 1].set_title(f'PCA construction with l = {l}')
    ax[1, 0].set_title(f'PCA construction with l = 1')
    ax[1, 1].set_title(f'PCA construction with l = 10')
    ax[2, 0].set_title(f'PCA construction with l = 50')
    ax[2, 1].set_title(f'PCA construction with l = 250')
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    plt.show()
mnist_train, X_mean = dataset(training_file)
# X_mean = np.reshape(X_mean, (28, 28))
cov_X = np.cov(mnist_train)
eigen_value_X, eigen_vector_X = np.linalg.eigh(cov_X)
l = int(input('please enter the l\'th largest principal component you want:'))
cmd = input('Do you want to generate the mean of X, too? Type yes or no:')
if (cmd == 'yes'):
    mean_visualization(X_mean)
sample_index = np.random.choice(2000)
X = mnist_train[:, sample_index]
# print(X.shape)

eigen_visualization(np.reshape(eigen_vector_X[:, -l], (28, 28)), eigen_value_X[-l], l)
PCA_visualization(X, l)
average_error_l = compute_reconstruction_error(mnist_train, l)
average_error_1 = compute_reconstruction_error(mnist_train, 1)
average_error_10 = compute_reconstruction_error(mnist_train, 10)
average_error_50 = compute_reconstruction_error(mnist_train, 50)
average_error_250 = compute_reconstruction_error(mnist_train, 250)
print(f'average error of  l = {l}, {average_error_l}')
print(f'average error of  l = {1}, {average_error_1}')
print(f'average error of  l = {10}, {average_error_10}')
print(f'average error of  l = {50}, {average_error_50}')
print(f'average error of  l = {250}, {average_error_250}')




    