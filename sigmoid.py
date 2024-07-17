import numpy as np

def naive_sigmoid(x, gain, mid):
    return 1.0 / (1.0 +np.exp((mid - x) * gain))

def scaled_sigmoid(x, gain, mid):
    min = naive_sigmoid(0.0, gain, mid)
    max = naive_sigmoid(1.0, gain, mid)
    s = naive_sigmoid(x, gain, mid)
    return (s - min) / (max - min)

def naive_inverse_sigmoid(x, gain, mid):
    min = naive_sigmoid(0.0, gain, mid)
    max = naive_sigmoid(1.0, gain, mid)
    s = naive_sigmoid(x, gain, mid)
    a = (max - min) * x + min
    return np.log(1.0 / a - 1.0)

def scaled_inverse_sigmoid(x, gain, mid):
    #if x < 0.00001 or x >= 0.99999:
    #    return x

    min = naive_inverse_sigmoid(0.0, gain, mid)
    max = naive_inverse_sigmoid(1.0, gain, mid)
    s = naive_inverse_sigmoid(x, gain, mid)
    return (s - min) / (max - min)
