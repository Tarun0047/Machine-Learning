import numpy as np
from sklearn import linear_model
import pandas as pd
import math
from word2number import w2n

df=pd.read_csv("homeprices.csv")

df.bedrooms.median()
df.bedrooms=df.bedrooms.fillna(df.bedrooms.median())

model=linear_model.LinearRegression()
model.fit(df.drop('price',axis='columns'),df.price)

model.coef_
model.intercept_
