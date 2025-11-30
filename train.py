import pandas as pd
import csv
from predict import linearRegression

t0 = 0.0
t1 = 0.0

db = pd.read_csv("./data.csv")


# db["km"] = db["km"] # Normalize
# db["price"] = db["price"]

learning_rate = 1e-12
iterations = 40000


def train():
    global t0, t1


    for _ in range(iterations):
        sum_t0 = 0.0
        sum_t1 = 0.0

        for mile, price in zip(db["km"], db["price"]):
            error = (linearRegression(mile, t0, t1) - price)
            sum_t0 += error
            sum_t1 += error * mile
        t0 -= learning_rate * 1/db.shape[0] * sum_t0
        t1 -= learning_rate * 1/db.shape[0] * sum_t1


    with open("thetaValue.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([t0, t1])