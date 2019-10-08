# import numpy
import numpy as np
#import pandas
import pandas as pd
# import scipy
import scipy.stats as sts
import scipy.special as spc
# import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#from matplotlib import cm
# import ternary plot lib
#import ternary
# import itertools
#import itertools
# improt math
#import math

#import sys

import design as doe

x = np.array([  [1.0, 0.326, 0.159],
                [0.9, 0.296, 0.144],
                [0.8, 0.252, 0.130],
                [0.7, 0.229, 0.112],
                [0.6, 0.194, 0.095],
                [0.5, 0.163, 0.080],
                [0.3, 0.098, 0.047]])

y = pd.DataFrame(x[:, 0], columns = ["Concentracao"],) 

x = pd.DataFrame(x[:, 1:], columns = ["619 nm", "663 nm"])

doe.log("X :", x)

doe.log("Y :", y)

Doe = doe.Factorial_Design(x, y, use_log = True)

Doe.plot()

Doe.error_plot(False, False)
"""

print("==========================================================")
print("==========================================================")
print("==========================================================")
print("==========================================================")
print("==========================================================")
print("==========================================================")

Doe2 = doe.Design(y, x, use_log=True)

Doe.plot()

Doe.error_plot(False, False)

"""