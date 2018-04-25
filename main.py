#!/usr/bin/env python


import scipy
import os
import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn import datasets


filename = "database.csv"

dataset = pd.read_csv(filename, usecols=[4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 19, 20, 23], dtype=str)

print(dataset)
