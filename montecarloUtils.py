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
    def init(self):
        pass

def main():
    mm = M(4,-0)
    print mm.matrix

if __name__ == "__main__":
    main()