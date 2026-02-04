import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing


housing = fetch_california_housing() # Example dataset; replace with actual dataset loading if needed

X_train, X_test, y_train, y_test = train_test_split(housing.data,housing.target, test_size=0.2,random_state=42)

# Standardize the features -> each feature contributes equally to the model
scaler = StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled,y_train)

joblib.dump(model, 'model.pkl')
print("Model trained and saved to model.pkl")


