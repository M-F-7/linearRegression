from predict import linearRegression, load_t
from train import db, train
import numpy as np


train()

t0, t1 = load_t()
print("t0: ", t0, "t1: ", t1)

test_mileages = [50000, 25000, 35000]
for m in test_mileages:
    print(f"Mileage: {m}, Estimated price: {linearRegression(m, t0, t1)}")



#BONUS

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(db["km"], db["price"], s=8)
plt.xlabel("mileage")
plt.ylabel("price")
plt.title("Price/Mileage")
plt.grid(True, alpha=0.3)
x_line = np.array([db["km"].min(), db["km"].max()])
y_line = linearRegression(x_line, t0, t1)
plt.plot(x_line, y_line, color="red", label=f"y = {t0:.1f} + {t1:.5f} x")
plt.show()