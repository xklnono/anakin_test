import numpy as np
import paddle.fluid as fluid
import os
from datetime import datetime



if __name__ == "__main__":

    random = np.random.RandomState(0)
    fp = open('input.txt', 'w')    
    #for i in range(270000):
    #for i in range(154587):
    #for i in range(147852): #plant
    for i in range(150528):  #animal
        a = random.uniform(0, 1)
        #print round(a, 2)
        fp.write('%f' % round(a, 2) + '\n')
    fp.close()
