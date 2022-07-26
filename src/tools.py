import pickle
import numpy as np


def print_cartesians(arr):
    """
    prints a 2-D numpy array in a nicer format
    """
    for a in arr:
        for i, elem in enumerate(a):
            if i == 0:
                print("{} ".format(int(elem)), end="")
            else:
                print("{} ".format(elem).rjust(3), end="")
        print(end="\n")


def write_pickle(data, fname="data.pickle"):
    with open(fname, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fname="data.pickle"):
    with open(fname, "rb") as handle:
        return pickle.load(handle)
