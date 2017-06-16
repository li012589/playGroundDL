import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from presentNN import genPic,save2Pic

BASE_DIR_PIC = './picDir/'

<<<<<<< HEAD
save2Pic(BASE_DIR_PIC,730000,1000)
=======
#save2Pic(BASE_DIR_PIC,67000,1000)

import imageio
images = []
for i in range(0,67000,1000):
    if i == 0 :
        continue
    images.append(imageio.imread(BASE_DIR_PIC+str(i)+'.png'))
imageio.mimsave(BASE_DIR_PIC+'combine.gif', images)
>>>>>>> 672713eed3bcf2b38e5487bb0da7e8f7f29869b4
