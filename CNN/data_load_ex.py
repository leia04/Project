import sys, os
sys.path.append(os.pardir)
from seul_prac.pneumonia_dataload import x_train_f

x = x_train_f
print(x.shape)
