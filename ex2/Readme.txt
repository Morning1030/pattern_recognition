Pattern Recognition Experiment2
EECS26 111060024

For ex2-1.py, ex2-2.py, and ex2-3.py, there is no need to input any arguments, the parameters are already fixed inside the program.
Therefore, to execute, simply type "python ${filename}.py" inside the terminal to run.
Remember that these files needs numpy library to run. Additionally, ex2-3 needs matplotlib to show the plot


The description of output each file:
ex2-1:
For (a) and (d), they share the same function that calculates the mean and biased variance for each feature. To get the full covariance
matrix in (d), it is diag(variance1, variance2, variance3). This function first output the mean vector, then the variance vector.
For (b), and (e), the output first output a mean vector, then the covariance matrix of sample data.

ex2-2:
This program outputs p_ml_a and p_ml_b that represents the maximum likelihood estimator of p for each dataset.
As the problem describes, dataset a has 1000 samples, and dataset b has 5000. They're both under prior = 0.7 with bernoulli distribution.

ex2-3:
This program shows two subplot for each dataset that corresponds to (a), (b) in ex2-3. It shows one figure for one dataset at a time, 
after you close the figure window, another one will be created with different dataset generated. This will repeat 10 times.


