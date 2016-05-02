import numpy as np
import scipy
class GaussFitter(object):
    def gaussian(self, height, center_x, center_y, width_x, width_y, rotation):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)

        rotation = np.deg2rad(rotation)
        center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
        center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

        def rotgauss(x,y):
            xp = x * np.cos(rotation) - y * np.sin(rotation)
            yp = x * np.sin(rotation) + y * np.cos(rotation)
            g = height*np.exp(
                -(((center_x-xp)/width_x)**2+
                    ((center_y-yp)/width_y)**2)/2.)
            return g
        return rotgauss

    def moments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y, 0.0


    def fitgaussian(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = scipy.optimize.leastsq(errorfunction, params)
        return p
            
import numpy
from numpy import median
def moments2(data,circle,rotate,vheight,estimator=median,**kwargs):
    """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output 
    a subset of the above.
    
    If using masked arrays, pass estimator=numpy.ma.median
    """
    total = numpy.abs(data).sum()
    Y, X = numpy.indices(data.shape) # python convention: reverse x,y numpy.indices
    y = numpy.argmax((X*numpy.abs(data)).sum(axis=1)/total)
    x = numpy.argmax((Y*numpy.abs(data)).sum(axis=0)/total)
    col = data[int(y),:]
    # FIRST moment, not second!
    width_x = numpy.sqrt(numpy.abs((numpy.arange(col.size)-y)*col).sum()/numpy.abs(col).sum())
    row = data[:, int(x)]
    width_y = numpy.sqrt(numpy.abs((numpy.arange(row.size)-x)*row).sum()/numpy.abs(row).sum())
    width = ( width_x + width_y ) / 2.
    height = estimator(data.ravel())
    amplitude = data.max()-height
    mylist = [amplitude,x,y]
    if numpy.isnan(width_y) or numpy.isnan(width_x) or numpy.isnan(height) or numpy.isnan(amplitude):
        raise ValueError("something is nan")
    if vheight==1:
        mylist = [height] + mylist
    if circle==0:
        mylist = mylist + [width_x,width_y]
        if rotate==1:
            mylist = mylist + [0.] #rotation "moment" is just zero...
            # also, circles don't rotate.
    else:  
        mylist = mylist + [width]
    return mylist

import numpy as N
def calculate_moments(frame):
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


    q = N.sqrt((m20 - m02) ** 2 + 4 * m11 ** 2)
    minor_axis = 2 ** 1.5 * N.sqrt(m20 + m02 + q)
    major_axis = 2 ** 1.5 * N.sqrt(m20 + m02 - q)
    angle = 0.5 * N.arctan2(2 * m11, m20 - m02)
    ellipticity = minor_axis / major_axis

    centroid = (m01, m10)

    return 8*m00/major_axis/minor_axis/N.pi, centroid, major_axis/4, minor_axis/4, angle

if __name__=="__main__":
    from numpy import *
    from numpy import random
    import MT
    #x=arange(640)
    #y=arange(480)
    X,Y=mgrid[:640, :480]
    frame=7*MT.gauss2d(X,Y, 320, 240, 50,70, 0)+0.0*random.normal(size=(640,480))
    gfit=GaussFitter()
    #print(gfit.fitgaussian(frame))
