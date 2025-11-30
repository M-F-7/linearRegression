import pandas as pd
import matplotlib.pyplot as plt
from predict import t0, t1, linearRegression


db = pd.read_csv("./data.csv")
learning_rate = 0.1

def train():
    global t0, t1

    sum_t0 = 0
    sum_t1 = 0

    error = 0
    for mile, price in zip(db["km"], db["price"]):
        error = (linearRegression(mile) - price)**2  
        sum_t0 += error
        sum_t1 += error * mile

    t0 -= learning_rate * 1/db.shape[0] * sum_t0
    t0 -= learning_rate * 1/db.shape[0] * sum_t1


print("theta0:", t0, "theta1:", t1)