import numpy as np
import MT
from Camera import Camera

class DummyCam(Camera):
    def __init__(self, res=(640,480), Nframes=10):
        x=np.linspace(-10,10, res[0])
        y=np.linspace(-10,10, res[1])
        Y,X=np.meshgrid(y,x)
        Ypos=np.linspace(-3,3,Nframes/2)
        Ypos=np.hstack([Ypos, Ypos[::-1]])
        #sclX=ones(Nframes)
        #gauss2d=100*gauss(X*sclX[:,np.newaxis, np.newaxis], [1, 0,0.2])*gauss(Y, [1, 0,0.2])
        self.dat=np.empty((Nframes, res[0], res[1]))
        for k in range(Nframes):
            if k<4:
                g2d=0
            else:
                g2d=200*MT.gauss2d(X,Y, 4, 4+0*Ypos[k],0.5,0.5,0)
            self.dat[k]=1*abs(np.random.normal(size=(res[0], res[1]) ) ) + g2d        #self.dat=self.dat.astype('u4')

        self.nextInd=0

    def query_frame(self):
        out=self.dat[self.nextInd]
        self.nextInd+=1
        if self.nextInd==self.dat.shape[0]:
            self.nextInd=0
        self.frame=out
        return out
