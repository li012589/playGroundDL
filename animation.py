import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from presentNN import genPic,save2Pic

BASE_DIR_PIC = './picDirPG/'
PASS0 = False
MAX = 300
STEP = 1
#save2Pic(BASE_DIR_PIC,67000,1000)

import imageio
images = []
for i in range(0,MAX,STEP):
    if i == 0 and PASS0:
        continue
    images.append(imageio.imread(BASE_DIR_PIC+str(i)+'.png'))
imageio.mimsave(BASE_DIR_PIC+'combine.gif', images)
