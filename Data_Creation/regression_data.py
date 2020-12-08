# imports
import numpy as np

# create arrays
inp = []
out = []


def create(val, inp, out):
    for i in range(val):
        inp.append(i)
        out.append(2*i + 1)

    inp = np.array(inp).reshape([-1, 1])
    out = np.array(out).reshape([-1, 1])

    return inp, out


def data(values=100):
    return create(values, inp, out)
