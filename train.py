# Data Manipulation libraries
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
import joblib
##data,y = load_boston(return_X_y=True)
print("1")
df = pd.read_csv('./data/tp3_boston_data.csv')  
    
#df = pd.DataFrame(data)  # Load the dataset

df_x = df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'lstat']]
df_y = df[['medv']]
print("2")
scaler = StandardScaler()
scaler.fit(df_x)

df_x_scaled = scaler.transform(df_x)
df_x_scaled = pd.DataFrame(df_x_scaled, columns=df_x.columns)
X_train, X_test, Y_train, Y_test = train_test_split(df_x_scaled, df_y, test_size = 0.33, random_state = 5)
print("3")
mlp = MLPRegressor(hidden_layer_sizes=(60), max_iter=10000)
mlp.fit(X_train, Y_train.values.ravel())
Y_predict = mlp.predict(X_test)
print("4")
#Saving the machine learning model to a file
joblib.dump(mlp, "./model/rf_model.pkl")