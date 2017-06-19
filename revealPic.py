import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from presentNN import genPic,save2Pic
import math

BASE_DIR_PIC = './picDir/'
BASE = 0
MAX = 6000
STEP = 1000
THETA1 = 0
THETA2 = 10*2*math.pi/360

save2Pic(BASE_DIR_PIC,BASE,MAX,STEP,THETA1)
save2Pic(BASE_DIR_PIC,BASE,MAX,STEP,THETA2)