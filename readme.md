# TODO:
A. implement logistic regression (lec 4)

B. implement discriminant analysis (lec 5)*

## Steps:
1. Implement logistic regression and LDA as python classes. Use a constructor for the class to initialize model parameters and attributes, and define other important properties.


2. Each class should have two functions:
    - fit (X, y) takes in the traiing data as well as hyperparameters (learning rate/number of iterations) as input. Train model by modifying model params.
    - predict (X) takes in the series of input points and outputs predictions. ie. 1 or 0 by using 0.5 threshold value on predicitons.


3. Define evaluate_acc function to evaluate model accuracy. This takes in data points (X), true labels (y), and target labels (y') as input and should output the accuracy score


4. Implement a script for k-fold cross validation


## Run Experiments:
Note: use 5-fold cross validation to estimate performacne of all experiemnts and evaulte performnce using accuracy NOT entropy function.

### At a minimum:
1. Test different leraning rates for logistic regression
2. Compare runtime and accuracy of LDA on both datasets.
3. For wine dataset try to find a new subset of features and/or additional features to improve the accuracy. (e.g. interaction terms?)
4. Go above and beyond for bonus points (e.g. investigate stopping criterea for grad. descent. in log. regression,analyze the performance tradoeoffs of different implementations for LDA, or develop automated approach to select a good subset of features. AND report on this is writeup!
