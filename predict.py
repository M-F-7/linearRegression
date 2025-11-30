import csv

def load_t():
    with open("thetaValue.csv", "r") as f:
        t0, t1 = map(float, next(csv.reader(f)))
    return t0, t1


def linearRegression(mileage, t0=0, t1=0) :
    estimatePrice =  t0 + (t1 * mileage)
    return estimatePrice