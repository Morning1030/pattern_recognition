import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(N):
    x = np.random.uniform(0, 10, N)
    epsilon = np.random.normal(0, 1, N)
    y = 3 * np.sin(0.8 * x + 2) + epsilon
    return x, y
# generate ten times
def generate_x_poly(order, training_set):
    N = training_set.shape[0]
    X = np.ones((N, 1))
    for d in range(1, order + 1):
        X = np.hstack((X, (training_set.reshape(-1, 1)) ** d))
    return X
def calc_error(y_predict, y_val):
    # use SSE
    error = np.sum((y_val - y_predict) ** 2)
    return error

data_split_ratio = 0.8
for i in range(10):
    dataset_x, dataset_y = generate_dataset(100)
    # split dataset
    split_index = np.int32(dataset_x.shape[0] * data_split_ratio)
    x_train = dataset_x[:split_index]
    # print(x_train)
    y_train = dataset_y[:split_index]
    x_val = dataset_x[split_index:]
    y_val = dataset_y[split_index:]
    train_error = []
    val_error = []
    # define x axis
    x_plot = np.linspace(0, 10, 500)
    # plot the fitting curve and error calculation for each dataset
    figure, ax = plt.subplots(1, 2)
    ax[0].plot(x_train, y_train, 'b+', label='training samples')
    
    # k  = 1, 3, 5, 7, 9
    for order in range(1, 10, 2):
        # train to get W
        X_train_poly = generate_x_poly(order, x_train)
        W = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
        # plot the fitting polynomial
        X_plot_poly = generate_x_poly(order, x_plot)
        Y_plot_predict = X_plot_poly @ W
        ax[0].plot(x_plot, Y_plot_predict, label=f'Order {order}')
        
        # calculate validation error and plot
        X_val_poly = generate_x_poly(order, x_val)
        Y_val_predict = X_val_poly @ W
        Y_train_predict = X_train_poly @ W
        val_error.append(calc_error(Y_val_predict, y_val))
        print(val_error)
        train_error.append(calc_error(Y_train_predict, y_train))
        print(train_error)
    ax[1].plot(range(1, 10, 2), val_error, label='validation')
    ax[1].plot(range(1, 10, 2), train_error, label='training')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[1].set_xlabel('order')
    ax[1].set_ylabel('SSE')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid(True)
    ax[1].grid(True)
    plt.suptitle(f'Dataset {i}')
    plt.show()
        

        
        
    
    