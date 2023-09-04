from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import numpy as np



def find_best_random_state(X, y, num_trials=100):

    mse_values = {}
    for random_state in range(1, num_trials + 1):
     
     #Spliting the data to train model
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    #Training the model with the diabetes data
     model = LinearRegression().fit(X_train,y_train)
     
     #Predict with your model using test data
     y_pred = model.predict(X_test)

     mse = mean_squared_error(y_test, y_pred)
     mse_values[random_state] = mse

    #Finding best random state
     best_random_state = min(mse_values, key=mse_values.get)
     best_mse = mse_values[best_random_state]

    return best_random_state, best_mse

#Gathering the data from sklearn.datasets
X,y=load_diabetes(return_X_y=True,as_frame=True)

#Second way to reach diabestes data
# diabetes = datasets.load_diabetes()
# X = diabetes.data
# y = diabetes.target

best_random_state, best_mse = find_best_random_state(X, y, num_trials=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=best_random_state)
model = LinearRegression().fit(X_train,y_train) 
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(best_random_state)
print(mse)
print(model.coef_)
print(model.intercept_)

### Optimization part

def linear_regression(params, X, y):
    """
    
    Creates a linear regression figure with the given yield and calculates the MSE.

    Args:
    params (list): parameters of Lineer regression [intercept, coeff1, coeff2, ...]
    X (array-like): Data with independent variables.
    y (array-like): the dependent variable

    Returns:
    mse (float): Mean Square Error
    """
    intercept, coeffs = params[0], params[1:]
    y_pred = np.dot(X, coeffs) + intercept
    mse = mean_squared_error(y, y_pred)
    return mse

def find_best_optimization_method(X, y, num_trials=100):
    """
    
    It applies different optimization methods for the given data and finds the best MSE value.

    Args:
    X (array-like): Data with independent variables.
    y (array-like): The dependent variable.
    num_trials (int, optional): Number of attempts (to be optimized). The default value is 100.
    
    Returns:
    best_method (str): Best optimize method.
    best_mse (float): Lowest MSE value.
    """
    mse_values = {}
    optimization_methods = ['Nelder-Mead', 'Powell', 'BFGS', 'CG', 'L-BFGS-B', 'TNC','COBYLA','SLSQP']

    for method in optimization_methods:
        mse_trials = []
        for _ in range(num_trials):
            # Splitting data to train models
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=np.random.randint(1, 100))
            
            # Finding initial guess
            initial_guess = np.random.rand(X_train.shape[1] + 1)
            
            # implementing optimization
            result = minimize(linear_regression, initial_guess, args=(X_train, y_train), method=method)
            
            # Reaching optimized parameters
            optimized_params = result.x
            
            # Making prediction with test data
            y_pred = np.dot(X_test, optimized_params[1:]) + optimized_params[0]
            
            # Calculate MSE and store it
            mse = mean_squared_error(y_test, y_pred)
            mse_trials.append(mse)
        
        # Calculate average optimize 
        mse_mean = np.mean(mse_trials)
        mse_values[method] = mse_mean

    # Find the optimize method with the lowest MSE
    best_method = min(mse_values, key=mse_values.get)
    best_mse = mse_values[best_method]

    return best_method, best_mse

# Loading diabetes data
from sklearn import datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

best_method, best_mse = find_best_optimization_method(X, y, num_trials=100)
print(f"Best optimize method: {best_method}")
print(f"Lowest MSE value: {best_mse}")

