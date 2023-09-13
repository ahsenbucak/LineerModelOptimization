import pandas as pd
# Loading diabetes data
csv_file="diabetes.csv"
df= pd.read_csv(csv_file)
df.drop('Outcome', axis=1, inplace=True)
print(df)

X = df[['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]  
y = df['Glucose']

print(X)
print(y)