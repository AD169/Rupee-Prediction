

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import datetime
from matplotlib import style

style.use("ggplot")

file = r"C:\Users\AD\Desktop\USD_INR Historical Data.csv"
df = pd.read_csv(file)

date = pd.to_datetime(df["Date"])
df.index = date

df = df.drop(["Change %","Date"],axis = 1)
df.dropna(inplace=True)

X = df.drop("Price", axis = 1)
y = df["Price"]


pipe1 = Pipeline([('scalar',StandardScaler()) , ('reg1',LinearRegression())])

pipe2 = Pipeline([('scalar',StandardScaler()) , ('reg2',Ridge())])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 21)

pipe1.fit(X_train,y_train)
pipe2.fit(X_train,y_train)

acu1 = pipe1.score(X_test,y_test)
acu2 = pipe2.score(X_test,y_test)
print(acu1,acu2)

y.plot(label = "Price",linewidth = 1)

df["Open"].plot(linewidth = 1)

plt.legend()

plt.show()


