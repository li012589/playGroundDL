import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from presentNN import genPic,save2Pic
import math

BASE_DIR_PIC = './picDir/'
PASS0 = True
MAX = 6000
STEP = 1000
THETA1 = 0
THETA2 = 10*2*math.pi/360
#save2Pic(BASE_DIR_PIC,67000,1000)

import imageio
def saveAnimation(BASE_DIR,maxstep,step,theta,pass0):
    images = []
    for i in range(0,maxstep,step):
        if i == 0 and pass0:
            continue
        images.append(imageio.imread(BASE_DIR+str(i)+"theta="+str(theta)+'.png'))
    imageio.mimsave(BASE_DIR+'combine_theta'+str(theta)+'.gif', images)

if __name__ == '__main__':
    saveAnimation(BASE_DIR_PIC,MAX,STEP,THETA1,PASS0)
    saveAnimation(BASE_DIR_PIC,MAX,STEP,THETA2,PASS0)
