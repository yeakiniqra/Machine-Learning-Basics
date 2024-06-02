import pandas as pd
import math
import numpy as np
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
print(df)

# Data preprocessing
median_bedrooms = math.floor(df.bedrooms.median())
print(median_bedrooms)

df.bedrooms = df.bedrooms.fillna(median_bedrooms)
print(df)

# Train the model
# y = m1x1 + m2x2 + b (m1, m2 are coefficients)
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price) # independent variables: area, bedrooms, age and target/dependant variable: price

print(reg.coef_) # m1, m2
print(reg.intercept_) # b

# Predict the price of a home with 3000 sqft area, 3 bedrooms, 40 years old
print(reg.predict([[3000, 3, 40]]))

# Predict the price of a home with 2500 sqft area, 4 bedrooms, 5 years old
print(reg.predict([[2500, 4, 5]]))