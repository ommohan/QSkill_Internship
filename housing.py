To Predict the price of a house based on features such as size, location, and number

from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler




california_housing = fetch_california_housing(as_frame=True)
df = california_housing.frame
X = california_housing.data
y = california_housing.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


normalization = StandardScaler()
X_train_scaled = normalization.fit_transform(X_train)
X_test_scaled = normalization.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)


from sklearn.metrics import mean_squared_error,mean_absolute_error

y_pred = lin_reg.predict(X_test_scaled)

print("Mean Absolute error:",mean_absolute_error(y_test,y_pred))
print("Mean squared error:", mean_squared_error(y_test,y_pred))
print(df.head())

print(y_pred[:6])
