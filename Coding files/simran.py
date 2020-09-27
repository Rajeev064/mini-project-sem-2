import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from math import sqrt

file = "MARUTI.csv"
df = pd.read_csv(file)

df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
# df.index = df['Date']

# plt.figure(figsize=(16, 8))
# plt.plot(df['Open'], df['Close'], label='Close Price history')

x = df[["Open", "High", "Low"]]
y = df["Close"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20)

reg = LinearRegression()
reg.fit(x_train, y_train)

y_predicted = reg.predict(x_test)
rmse = sqrt(mean_squared_error(y_test, y_predicted))

plt.plot(y_predicted, y_test)

print("Root mean squared error", rmse)
print("Score", reg.score(x_test, y_test))
plt.show()