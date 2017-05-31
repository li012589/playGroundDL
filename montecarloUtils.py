import numpy as np

class M:
    def __init__(self,size,init):
        self.size = size
        if init == 1:
            self.matrix = np.ones(size*size)
        elif init == -1:
            self.matrix = np.ones(size*size)
            self.matrix[:] = -1
        else:
            self.matrix = np.random.random(tuple([size*size]))
            self.matrix[self.matrix<0.5] = -1
            self.matrix[self.matrix>=0.5] = 1
    def calculateNo(self,i,j):
        return j*self.size+i
    def init(self):
        self.adjacentTab = {}
        for i in range(self.size):
            for j in range(self.size):
                if i-1 < 0:
                    r = self.size-1
                else:
                    r = i-1
                if i+1 >= self.size:
                    l = 0
                else:
                    l = i+1
                if j-1 <0:
                    d = self.size-1
                else:
                    d = j-1
                if j+1 >= self.size:
                    u = 0
                else:
                    u = j+1
                self.adjacentTab[self.calculateNo(i,j)] = [self.calculateNo(r,j),self.calculateNo(l,j),self.calculateNo(i,u),self.calculateNo(i,d)]

def main():
    mm = M(4,-0)
    mm.init()
    print mm.matrix
    print mm.adjacentTab[(0)]

if __name__ == "__main__":
    main()