
from sklearn.datasets import load_boston
import pandas as pd

data,y = load_boston(return_X_y=True)

print(type(data))
#data.columns
#print(data.columns)

df = pd.DataFrame(data)

print(df[0])