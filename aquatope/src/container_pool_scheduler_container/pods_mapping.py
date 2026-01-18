import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import csv
import joblib

# Sample data
X = np.array([
    [563135, 30],
    [563135, 60],
    [563135, 90],
    [563135, 120],
    [563135, 150],
])

Y = np.array([
    [2, 10],
    [9, 22],
    [13, 38],
    [19, 58],
    [24, 73],
])

# Polynomial regression model
model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("reg", LinearRegression())
])

model.fit(X, Y)

joblib.dump(model, "polynomial_regression_model.joblib")
# model = joblib.load("polynomial_regression_model.joblib")


# Predict
x_new = np.array([
  [563135, 109],
  [563135, 107],
  [563135, 97],
  [563135, 92],
  [563135, 124],
  [563135, 111],
  [563135, 124],
  [563135, 106],
  [563135, 94],
  [563135, 68],
  [563135, 74],
  [563135, 90],
  [563135, 70],
  [563135, 82],
  [563135, 111],
  [563135, 105],
  [563135, 82],
  [563135, 90],
  [563135, 87],
  [563135, 65],
])

x_new  = x_new[np.argsort(x_new[:, 1])[::-1]]
results = model.predict(x_new)


for i, x in enumerate(x_new):
    a_pred, b_pred = results[i]
    print(f"({i}, {x.tolist()}): Predicted a={round(a_pred)}, b={round(b_pred)}")

csv_file = "pods_mapping.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["input_size", "rps", "a", "b"])

    for i, x in enumerate(x_new):
        a_pred, b_pred = results[i]
        writer.writerow([
            x[0],                # rps
            x[1],                # input_size
            round(a_pred),       # a
            round(b_pred),       # b
        ])

print(f"Saved predictions to {csv_file}")

"""
(0, [563135, 109]): Predicted a=17, b=49
(1, [563135, 107]): Predicted a=17, b=48
(2, [563135, 97]): Predicted a=15, b=43
(3, [563135, 92]): Predicted a=14, b=40
(4, [563135, 124]): Predicted a=20, b=58
(5, [563135, 111]): Predicted a=17, b=50
(6, [563135, 124]): Predicted a=20, b=58
(7, [563135, 106]): Predicted a=17, b=48
(8, [563135, 94]): Predicted a=14, b=41
(9, [563135, 68]): Predicted a=10, b=27
(10, [563135, 74]): Predicted a=11, b=30
(11, [563135, 90]): Predicted a=14, b=39
(12, [563135, 70]): Predicted a=10, b=28
(13, [563135, 82]): Predicted a=12, b=35
(14, [563135, 111]): Predicted a=17, b=50
(15, [563135, 105]): Predicted a=16, b=47
(16, [563135, 82]): Predicted a=12, b=35
(17, [563135, 90]): Predicted a=14, b=39
(18, [563135, 87]): Predicted a=13, b=37
(19, [563135, 65]): Predicted a=9, b=26
"""