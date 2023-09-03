from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Diyabet verilerini yükle
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Verileri eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Lineer Regresyon modelini oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)


print(model.coef_)
print(model.intercept_)

# Test verileri üzerinde tahmin yap
# y_pred = model.predict(X_test)

# Hata ölçümü
# mse = mean_squared_error(y_test, y_pred)
# print(f"Ortalama Kare Hata (MSE): {mse}")