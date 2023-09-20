import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

csv_file="NewHousingData.csv"
df= pd.read_csv(csv_file)
X = df.drop(['PRICE'], axis = 1)
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
model = LinearRegression().fit(X_train,y_train)

initial_features = np.random.rand(X_train.shape[1] + 1) 
print(X_train.shape[1])
print(initial_features)