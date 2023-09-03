import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



#Gathering the data from sklearn.datasets
X,y=load_diabetes(return_X_y=True,as_frame=True)


#Spliting the data to train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=1)

#Training the model with the diabetes data
model = LinearRegression().fit(X_train,y_train)

#print(model.coef_)
#print(model.intercept_)


