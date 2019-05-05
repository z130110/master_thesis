import numpy as np 
import matplotlib.pyplot as plt
from sklearn import manifold


path_ = "gp_ln/en_zh/"
file_name = "top10k_en.npy"


load_data = np.load(path_ + file_name)

print(load_data)