import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from presentNN import genPic,save2Pic

BASE_DIR_PIC = './picDir/'
MAX = 730000
STEP = 1000

save2Pic(BASE_DIR_PIC,MAX,STEP)