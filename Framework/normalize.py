# imports
import numpy as np

# normalizing a numpy array and returning the normalized 
def normalize(arr):
    mini = np.min(arr)
    maxi = np.max(arr)

    diff = maxi - mini
    arr = (arr - mini) / diff

    return arr, mini, diff
