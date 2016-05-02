#coding: utf8
import numpy as np
from scipy import optimize as opt
from MT import gauss, gauss2d
import pyqtgraph as pg

def _calculate_moments(frame):
    """Calculate the moments"""
    # From Bullseye
    y, x = np.mgrid[:frame.shape[0], :frame.shape[1]]
    m00 = frame.sum() #or 1.0
    m10 = (frame * x).sum() / m00
    m01 = (frame * y).sum() / m00
    dx, dy = x - m10, y - m01
    m20 = (frame * dx ** 2).sum() / m00
    m02 = (frame * dy ** 2).sum() / m00
    m11 = (frame * dx * dy).sum() / m00
    return m00, m10, m01, m20, m02, m11

class NearPeakConstrainer(object):
    def __init__(self, initialGuess, threshold):
        self.initialGuess=100*initialGuess/initialGuess.max()
        self.threshold=threshold
        
    def __call__(self, pVec):
        x0=int(pVec[0])
        y0=int(pVec[1])
        try:
            val= self.initialGuess[x0,y0]-self.threshold
        except IndexError:
            val=-10;
        print("constraint val: {}".format(val))
        return val

resObj=None


class Gauss2dModel(object):
    resObj=None;
    def __init__(self, Nx, Ny, targetFrame=None, initialGuess=None):
        x=np.arange(Nx)#linspace(0,1,arr2d.shape[0])
        y=np.arange(Ny)#linspace(0,1,arr2d.shape[0])
        self.Y,self.X=np.meshgrid(y,x)
        self.x0=Nx/2
        self.y0=Ny/2
        self.sig1_0=Nx/2.
        self.sig2_0=Ny/2.
        self.theta_0=0.
        self.height0=200.

        self.targetFrame=targetFrame
        self.initialGuess=initialGuess
        self.Nx, self.Ny =Nx, Ny

    def getFrame(self, height0=None, x0=None, y0=None, sig1_0=None, sig2_0=None, theta0=None):
        if x0 is None:
            x0=self.x0
        if y0 is None:
            y0=self.y0
        if sig1_0 is None:
            sig1_0=self.sig1_0
        if sig2_0 is None:
            sig2_0=self.sig2_0
        if theta0 is None:
            theta0=self.theta0
        if height0 is None:
            height0=self.height0
        return height0*gauss2d(self.X, self.Y, x0, y0, sig1_0, sig2_0, theta0 )

    def evalXY(self, x, y):
        return self.height0*gauss2d(x, y, self.x0, self.y0, self.sig1_0, self.sig2_0, self.theta0 )
        
    def getFramePar(self, p):
        return self.getFrame(*p)

    def setTargetFrame(self, targetFrame):
        self.targetFrame=targetFrame

    def getErr(self, p=None, bShow=False):
        if p is None:
            p=self.getP()
        guess=self.getFramePar(p)       #figure(2)
        #guess=gauss(X, [sqrt(p[0]), p[1], p[3]])*gauss(Y, [sqrt(p[0]), p[2], p[3]])
        err = ((guess-self.targetFrame)**2).sum()
        if bShow:
            pg.image(guess, title='guess') 
            pg.image(self.targetFrame, title='frame')
        #imshow(guess)
        #draw()
        #print("Params: {0}".format(p))
        if err==0:
            err=1e-12
        err=np.log(err)
        print("err, height: {}, {}".format(err, p[0]))
        return err;

    def getConstrFunc(self, threshold=50):
        def f(p0):
            x0=int(p0[1])
            y0=int(p0[2])
            try:
                val=self.evalXY(x0,y0)/self.height0*100-threshold
            except IndexError:
                val=-100;
            print("constraint val: {}".format(val))
            return val
        return f

    def setInitialPars(self, p0):
        self.height0=p0[0]; 
        self.x0=p0[1]; 
        self.y0=p0[2]; 
        self.sig1_0=p0[3]; 
        self.sig2_0=p0[4]; 
        self.theta0=p0[5]
        self.initialGuess=self.getFrame()

    def getBounds(self):
        bounds=[(5,1000), (0,self.Nx), (0, self.Ny), (2,self.Nx), (2,self.Nx), (None,None)]
        return bounds

    def getP(self):
        return [self.height0, self.x0, self.y0, self.sig1_0, self.sig2_0, self.theta0]

    def fit(self, maxIters=200):
        #opt.fmin(func, x0, args=(), xtol=0.0001, ftol=0.0001, maxiter=None, maxfun=None, full_output=0, disp=1, retall=0, callback=None)
        #pg.image(self.initialGuess, title='initial')
        print("Initial err:{}".format(self.getErr(self.getP())))
        if maxIters==0:
            return self.getP()
        #p= opt.fmin_bfgs(f, p0, maxiter=maxIters)
        pkConstr=self.getConstrFunc(50)
        cons=[dict(type='ineq', 
            fun=pkConstr
            )]
        res= opt.minimize(self.getErr, self.getP(),  constraints=cons,options={'maxiter':maxIters, 'disp':True}, bounds=self.getBounds(), tol=1e-8)
        p=res.x
        self.resObj=res
        return p

    def getLastFit(self):
        if self.resObj is not None:
            return self.getFramePar(self.resObj.x)
        else:
            return None

    def fit_by_moments(self):
        """Calculate the moments"""
        # From Bullseye
        frame=self.targetFrame
        y, x = np.mgrid[:frame.shape[0], :frame.shape[1]]
        m00 = frame.sum() #or 1.0
        m10 = (frame * x).sum() / m00
        m01 = (frame * y).sum() / m00
        dx, dy = x - m10, y - m01
        m20 = (frame * dx ** 2).sum() / m00
        m02 = (frame * dy ** 2).sum() / m00
        m11 = (frame * dx * dy).sum() / m00


        q = np.sqrt((m20 - m02) ** 2 + 4 * m11 ** 2)
        minor_axis = 2 ** 1.5 * np.sqrt(m20 + m02 + q)
        major_axis = 2 ** 1.5 * np.sqrt(m20 + m02 - q)
        angle = 0.5 * np.arctan2(2 * m11, m20 - m02)
        #ellipticity = minor_axis / major_axis

        #centroid = (m01, m10)

        return 8/np.pi*m00/major_axis/minor_axis, m01, m10, major_axis/4, minor_axis/4, (angle+np.pi)%(2*np.pi) -np.pi


def doFit(frame, background_percentile=10, num_crops=1, crop_radius=1.5):
    bw = (len(frame.shape) == 2)
    if not bw:
        # Use standard NTSC conversion formula
        frame = np.array(
            0.2989 * frame[..., 0]
            + 0.5870 * frame[..., 1]
            + 0.1140 * frame[..., 2])

    # Calibrate the background
    print("frm.mean:{}".format(frame.mean()))
    background = np.percentile(frame, background_percentile)
    print('bg:{}, mean:{}'.format(background, np.mean(frame)))
    frame -= background
    np.clip(frame, 0.0, frame.max(), out=frame)

    m00, m10, m01, m20, m02, m11 = _calculate_moments(frame)

    if 1:
        bc, lc = 0, 0
        for count in range(num_crops):
            include_radius, dlc, dbc, drc, dtc, frame = _crop(frame,
                crop_radius, m00, m10, m01, m20, m02, m11)
            lc += dlc
            bc += dbc

            # Recalibrate the background and recalculate the moments
            new_bkg = np.percentile(frame, background_percentile)
            frame -= new_bkg
            print('newbg:{}, mean:{}'.format(new_bkg, frame.mean()))
            background += new_bkg
            np.clip(frame, 0.0, frame.max(), out=frame)

            m00, m10, m01, m20, m02, m11 = _calculate_moments(frame)

    print(m00, m10, m01, m20, m02, m11)
    m10 += lc
    m01 += bc

    # Calculate Gaussian boundary
    q = np.sqrt((m20 - m02) ** 2 + 4 * m11 ** 2)
    outD=dict(
        major_axis = 2 ** 1.5 * np.sqrt(m20 + m02 + q),
        minor_axis = 2 ** 1.5 * np.sqrt(m20 + m02 - q),
        angle = np.degrees(0.5 * np.arctan2(2 * m11, m20 - m02)),
        centroid = (m10, m01),
        baseline = background,
        include_radius = include_radius
    )

    outD['ellipticity'] = outD['minor_axis'] / outD['major_axis']
    outD['frame']=frame

    return outD

def doFitCompl(frame):
    outD=doFit(frame, background_percentile=10, num_crops=1, crop_radius=1.5)
    centroid=outD['centroid']
    width=outD['major_axis']

    p,arr=fit_2d_gaussian(frame, x0=centroid[0], y0=centroid[1], sig_1_0=outD['major_axis'], sig_2_0=outD['minor_axis'], theta0=outD['angle']/180*np.pi, height0=frame.max(), ret_fit=True)
    return p, arr





def _crop(frame, crop_radius, m00, m10, m01, m20, m02, m11):
    """crop based on 3 sigma region"""
    w20 = crop_radius * 4 * np.sqrt(m20)
    w02 = crop_radius * 4 * np.sqrt(m02)
    include_radius = np.sqrt((w20 ** 2 + w02 ** 2) / 2)
    w02 = max(w02, 4)
    w20 = max(w20, 4)
    lc = int(max(0, m10 - w20))
    bc = int(max(0, m01 - w02))
    tc = int(min(frame.shape[0], m01 + w02))
    rc = int(min(frame.shape[1], m10 + w20))
    frame = frame[bc:tc, lc:rc]
    return include_radius, lc, bc, rc, tc, frame


if __name__=="__main__":
    from DummyCam import DummyCam
    dc=DummyCam()
    import pyqtgraph as pg
    from numpy import *
    frame=dc.get_frame()
    pg.image(frame)

    par, arr=doFitCompl(frame)
