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

x = np.array([  [1.0, 0.326],
                [0.9, 0.296],
                [0.8, 0.252],
                [0.7, 0.229],
                [0.6, 0.194],
                [0.5, 0.163],
                [0.3, 0.098]])

y = pd.DataFrame(x[:, 1], columns = ["Intensidade",])

x = pd.DataFrame(x[:, 0], columns = ["Concentracao"],) 

doe.log("X :", x)

doe.log("Y :", y)

Doe = doe.Design(x, y, use_log = True)

Doe.plot()