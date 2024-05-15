import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv("homeprices.csv")
print(df)

plt.scatter(df['area'], df['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Home Prices')
plt.show()


reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])

predicted_price = reg.predict(np.array([[5000]])) 
print(predicted_price)

d = pd.read_csv("areas.csv")
print(d.head(4))

p = reg.predict(d[['area']])  
d['prices'] = p


d.to_csv("prediction.csv",index=False)

