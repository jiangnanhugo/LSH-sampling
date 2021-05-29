from jflt import fjlt
import numpy as np

A =np.random.rand(30,20)
k =5
p =0.2
print(fjlt(A, k, p).shape)