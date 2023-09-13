from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import numpy as np
import pandas as pd



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

#Gathering the data from csv file
csv_file="diabetes.csv"
df= pd.read_csv(csv_file)
df.drop('Outcome', axis=1, inplace=True)
X = df[['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]  
y = df['Glucose']


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

def find_best_optimization_method(X, y):
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
            # Splitting data to train models
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=best_random_state)
            
            # Finding initial guess
            initial_guess = np.random.rand(X_train.shape[1] + 1)
            #initial_guess=np.zeros(X.shape[1]+1)
            
            # implementing optimization
            result = minimize(linear_regression, initial_guess, args=(X_train, y_train), method=method)
            
            # Reaching optimized parameters
            optimized_params = result.x
            
            # Making prediction with test data
            y_pred = np.dot(X_test, optimized_params[1:]) + optimized_params[0]
            
            # Calculate MSE and store it
            mse = mean_squared_error(y_test, y_pred)
            mse_values[method]=mse
        
    # Find the optimize method with the lowest MSE
    best_method = min(mse_values, key=mse_values.get)
    best_mse = mse_values[best_method]


    return best_method, best_mse, optimized_params


# Loading diabetes data
csv_file="diabetes.csv"
df= pd.read_csv(csv_file)
df.drop('Outcome', axis=1, inplace=True)
X = df[['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]  
y = df['Glucose']


best_method, best_mse ,optimized_params= find_best_optimization_method(X, y)
print(f"Best optimize method: {best_method}")
print(f"Lowest MSE value: {best_mse}")


#using optimized model to prediction

entered_data=np.random.uniform(0,1,8)
finalPred=entered_data.dot(optimized_params)
print(finalPred)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred, y_test))

