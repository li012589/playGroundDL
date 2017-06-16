import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
basePath = './picDirPG/'
for i in range(0,4):
    with open(basePath + "NNZeroresult" + str(i), "rb") as fp:
        zeros = pickle.load(fp)
    with open(basePath + "NNOneresult" + str(i), "rb") as fp:
        ones = pickle.load(fp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(zeros[0], zeros[1], zeros[2],s=1, c='r', marker='o',alpha=0.1)
    ax.scatter(ones[0], ones[1], ones[2], s=1,c='b', marker='^',alpha=0.1)
    ax.set_xlabel('x_dot')
    ax.set_ylabel('theta_dot')
    ax.set_zlabel('theta')
    ax.text2D(0.05, 0.95, "red: right; blue: left", transform=ax.transAxes)
    fig.savefig(basePath+str(i)+'.png')
    plt.close(fig)
    #plt.show()