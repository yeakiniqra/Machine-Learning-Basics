import pandas as pd
import math
import numpy as np
from sklearn import linear_model
from word2number import w2n

df = pd.read_csv("hiring.csv")
print(df)

# Data preprocessing
median_test_score = math.floor(df['test_score(out of 10)'].median())
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)
print(df)

df['experience'] = df['experience'].apply(w2n.word_to_num)

# Train the model
# y = m1x1 + m2x2 + m3x3 + b (m1, m2, m3 are coefficients)
reg = linear_model.LinearRegression()
reg.fit(df[['test_score(out of 10)', 'experience', 'interview_score(out of 10)']], df['salary($)']) # independent variables: test_score, experience, interview_score and target/dependant variable: salary

print(reg.coef_) # m1, m2, m3
print(reg.intercept_) # b

# Predict the salary of a candidate with 10 test score, 5 years of experience, 7 interview score
print("Predicted salary of a candidate with 10 test score, 5 years of experience, 7 interview score: ")
print(reg.predict([[10, 5, 7]]))

# Predict the salary of a candidate with 8 test score, 3 years of experience, 8 interview score
print("Predicted salary of a candidate with 8 test score, 3 years of experience, 8 interview score: ")
print(reg.predict([[8, 3, 8]]))

