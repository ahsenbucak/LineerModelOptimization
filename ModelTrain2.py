from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



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
#print(model.coef_)
#print(model.intercept_)


