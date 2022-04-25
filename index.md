# Project 5

## Motivation

The following project compares the abilities of various regularization methods to model a data set which contains far more features than observations, along with heavily multicolineated features. A data set such as this one is very difficult to accurately model with OLS regression, as the noise of correlated features often leads to models being weak outside of the training set. Regularization, which refers to any technique designed to filter predictive features, is one main way to improve regression out-of-sample.

## Methods

The regularization techniques we are using are Lasso, Square Root Lasso, Ridge, SCAD, and Elastic Net. Each of these techniques works in a similar way; they attempt to reduce the magnitude of the coefficients of the OLS model without raising the residuals. This is done by adding some cost function, which is a function of the absolute value of the coefficients, to the residuals which OLS typically intends to reduce.

The main difference between these methods is the curve which dictates how large that cost function is at various magnitudes of the coefficient. Using an artificial data set of collinear variables, we will try to see which of these cost functions is most effective, using three metrics using 10-fold cross validation: the average root mean squared error of the the model out of sample, the euclidean distance between out final model and the true base model, and the number of coefficients which remain above 0 in the regularized model.

Each of these regularization techniques have a parameter, alpha, which essentially determines how much coefficients should be penalized; a higher alpha will result in a final model with more variables eliminated. SCAD has a second parameter, lambda, which also contributes to the shape of the penalty curve. We will use GridSearchCV, an SKLearn function, to identify the optimal value of these parameters for each technique. 

Our data set is made up of 200 observations of 1200 features. The correlation of each adjacent feature is \rho = .8 . The true value of each coefficient is also given, with only 27 true non-zero coefficients of the 1200.  Lastly, a significant random noise function was added to each feature.

While Ridge, Lasso, and Elastic Net are all built into SKLearn, we must construct our own versions of Square Root Lasso and SCAD, shown below.

```Python
from numba import njit
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

from math import ceil
from scipy import linalg
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

@njit
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part

@njit    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))

class SCAD(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=2,lam=1):
        self.alpha, self.lam = alpha, lam
  
    def fit(self, x, y):
        alpha = self.alpha
        lam   = self.lam

        @njit
        def scad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          return 1/n*np.sum((y-x.dot(beta))**2) + np.sum(scad_penalty(beta,lam,alpha))

        @njit  
        def dscad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          output = -2/n*np.transpose(x).dot(y-x.dot(beta))+scad_derivative(beta,lam,alpha)
          return output.flatten()
        
        
        beta0 = np.zeros((x.shape[1],1))
        output = minimize(scad, beta0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 50,'disp': False})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```

```Python
class SQRTLasso(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
  
    def fit(self, x, y):
        alpha=self.alpha
        @njit
        def f_obj(x,y,beta,alpha):
          n = x.shape[0]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.sqrt(1/n*np.sum((y-x.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
          return output
        @njit
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = (-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)
          return output.flatten()
        
        def objective(beta):
          return(f_obj(x,y,beta,alpha))
        
        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.ones((x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```

The following function was made in order to initialize the data set, fit each model using GridSearchCV, and extract our metrics of interest. 

```Python
def get_metrics(penalty, param_dict):
  
  beta_star = np.concatenate(([1]*7,[0]*25,[0.25]*5,[0]*50,[0.7]*15,[0]*1098))

  v = []
  for i in range(1200):
    v.append(0.8**i)

  mu = [0]*1200
  sigma = 3.5
  np.random.seed(123)
  x = np.random.multivariate_normal(mu, toeplitz(v), size=200)
  y = np.matmul(x,beta_star).reshape(-1,1) + sigma*np.random.normal(0,1,size=(200,1))
  grid = GridSearchCV(estimator = penalty(), cv = 10,scoring = 'neg_root_mean_squared_error', param_grid = param_dict)
  grid.fit(x, y)
  RMSE = grid.best_score_
  dist = np.linalg.norm(grid.best_estimator_.coef_ - beta_star)
  params = grid.best_estimator_.get_params()

  nonzeros = 0
  #nzarray = np.concatenate(([True]*7,[False]*25,[True]*5,[False]*50,[True]*15,[False]*1098))
  
  reshapes = [Ridge, Lasso]
  if penalty in reshapes:
    coefs = grid.best_estimator_.coef_.reshape(-1,1)
  else: 
    coefs = grid.best_estimator_.coef_
  for i in range(len(beta_star)):
    if (abs(coefs[i]) > .00001):
      nonzeros += 1
  print("RMSE: " + str(-RMSE))
  print("L2 distance: " + str(dist))
  print("Correctly picked nonzeros: " + str(nonzeros))

  return RMSE, dist, nonzeros, params, coefs
```

## Results

The following executions of our *get_metric* function were used to get our final results.

```Python
RMSE_SL, dist_SL, nonzeros_SL, params_SL, coefs_SL = get_metrics(SQRTLasso, {'alpha': np.linspace(.05, .25, 10)})
```

```Python
RMSE_Ridge, dist_Ridge, nonzeros_Ridge, params_Ridge, coefs_Ridge = get_metrics(Ridge, {'alpha': np.linspace(.0000000001, .001, 10)})
```

```Python
RMSE_Lasso, dist_Lasso, nonzeros_Lasso, params_Lasso, coefs_Lasso = get_metrics(Lasso, {'alpha': np.linspace(.05, .25, 10)})
```

```Python
RMSE_EN, dist_EN, nonzeros_EN, params_EN, coefs_EN = get_metrics(ElasticNet, {'alpha': np.linspace(.001, .01, 10)})
```

```Python
RMSE_SCAD, dist_SCAD, nonzeros_SCAD, params_SCAD, coefs_SCAD = get_metrics(SCAD, {'alpha': np.linspace(1.5, 5, 4), 'lam': np.linspace(.0000000001, .001, 4)})
```

| Regularization Technique  | Non-zero coefficients | RMSE | Euclidean Distance |
| ------------- | ------------- | ------------- | ------------- |
| Square Root Lasso  | 80  | 3.82  | 1.17  |
| Ridge  | 1200  | 5.95  | 3.00  |
| Lasso  | 109  | 3.94  | 3.72  |
| Elastic Net  | 374  | 3.89  | 2.96  |
| SCAD  | 1200  | 5.86  | 3.00  |

As shown, we can see that there is a large variance in the number of variables that are permitted by the penalty to remain in the final model. Ridge regression in particular does not reduce variables to zero, so all 1200 features are still represented. However, each method is relatively strong in terms of RMSE and Euclidean distance. Square Root Lasso appears to work best, especially in terms of Euclidean distance. On the other hand, SCAD and Ridge regression are less effective.




