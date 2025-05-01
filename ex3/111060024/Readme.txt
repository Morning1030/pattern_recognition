Pattern Recognition Experiment3
EECS26 111060024

For ex3-2.py, and ex3-3.py, there is no need to input any arguments, the parameters are already fixed inside the program.
Therefore, to execute, simply type "python ${filename}.py" inside the terminal to run.
Remember that these files needs numpy and matplotlib to run, install them before running.
ex3-2 and ex3-3 both relies on shared_dataset.py and visualize.py, which generates data and visualize with 2-D plot.

ex3-1 will ask about the reduced dimension l with PCA, and ask about whether to show the average image while showing the eigenvectors.
Input a integer less than 784 for l, and simply answer yes or no for the average showing question.



The description of output each file:
ex2-1:
This file outputs the eigenvector that corresponds to l'th largest eigenvector and the eigenvector w1, w2, w3, and w4. It also shows the average data
after asking the user. After that it shows the sampled data from the dataset and the PCA reconstruction with dimension = l, 1, 10, 50, and 250 at one plot.
At the terminal, it prints out the reconstruction error from each dimension.

ex2-2:
This program outputs the visualization of two plots for each dataset: the first one is the data with ground truth label, and the second one is the data with 
predicted label, along with the initial sample chosen. It first show for the dataset described in the problem, then the dataset described at (b)

ex2-3:
This program outputs the visualization of four plots for each dataset: the first one is the data with ground truth label, then the clustering result of K = 2,
K = 3, and K = 4. It first show for the dataset described in the problem, then the dataset described in (b).


