"""
Simple beam profiler gui
"""

from pylab import *
import numpy as np
#from morgTools.morgTools import gauss
import sys
from scipy import optimize as opt
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
#from beamprofile import doFit
import beamprofile as bp
from DummyCam import DummyCam



if 0:
    pygame.init()
    from pygame import camera as pgcamera
    pgcamera.init()
    cam = pgcamera.Camera("/dev/video2")#,(640,480))
    cam.start()

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph import dockarea as da


RES=(640,480)

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
area = da.DockArea()
win.setCentralWidget(area)
win.resize(1200,600)
win.setWindowTitle('beam profiling')

def getROICenter(roi):
    pos=roi.pos()
    size=roi.size()
    newCenter=(pos[0]+size[0]/2., pos[1]+size[1]/2. )
    return np.array(newCenter)

class MainArea(object):
    #flags
    bSubtractBG=False

    def __init__(self, frame=[[]], area=None):
        frame=np.array(frame)
        imageDck = da.Dock("Intensity", size=(500,400))
        gLayout=pg.GraphicsLayoutWidget()
        view1=gLayout.addViewBox()
        view1.setAspectLocked(True)
        img=pg.ImageItem(border='w')
        view1.addItem(img)
        #view=pg.ViewBox()
        #view.setAspectLocked(True)
        #gv.setCentralWidget(view)
        imageDck.addWidget(gLayout)
        # Contrast/color control
        hist = pg.HistogramLUTItem()
        #hist.setImageItem(img)
        gLayout.addItem(hist)
        area.addDock(imageDck)

        rectRoi=pg.RectROI((0,0), frame.shape, centered=True)
        view1.addItem(rectRoi)
        if area is None:
            area= da.DockArea()
        area.addDock(imageDck, 'left')

        self.rectRoi=rectRoi
        self.view=view1;
        self.gLayout=gLayout
        self.img=img
        self.dck=imageDck
        self.area=area

        self.updateFrame(frame)
        rectRoi.sigRegionChanged.connect(self.roiUpdated)

    def roiUpdated(self):
        self.subFrame=self.rectRoi.getArrayRegion(self.frame, self.img)

    def updateFrame(self, newFrame):
        if self.bSubtractBG:
            newFrame=self.subtractBG(newFrame)
        self.frame=newFrame
        self.img.setImage(newFrame)
        self.roiUpdated()

    def subtractBG(self, newFrame):
        if self.bg is None:
            self.bg=np.zeros(newFrame.shape)
        try:
            subbed=newFrame-self.bg
        except ValueError: #Not the right error
            self.bg=np.zeros(newFrame.shape)
            subbed=newFrame

        return subbed

    def setBG(self):
        self.bg=self.frame

    def resetCrop(self):
        self.rectRoi.setPos((0,0))
        self.rectRoi.setSize(self.frame.shape)

class Cropped(object):
    #Flags
    bUpdateProjections=True
    bAutoFit=False
    fitFrame=None
    oldCenter=np.zeros(2)
    bDoFullFit=True

    def __init__(self, frame, rectRoi, area=None):
        self.dck = da.Dock("Cropped", size=(500,400))
        self.ellipseRoi = pg.EllipseROI((RES[0]/4,  RES[1]/4), (RES[0]/2, RES[1]/2))
        self.ellipseRoi.setZValue(10)  # make sure ROI is drawn above image
        self.subImg=pg.ImageItem(border='w')    
        self.subWidget=pg.GraphicsLayoutWidget()
        self.subView=pg.ViewBox()
        self.subWidget.addItem(self.subView)
        self.subView.addItem(self.subImg)
        self.subView.addItem(self.ellipseRoi)
        self.dck.addWidget(self.subWidget)


        self.dckProjPlot=da.Dock("Projections")
        self.pltX=pg.PlotWidget()
        self.pltY=pg.PlotWidget()
        self.dckProjPlot.addWidget(self.pltX)
        self.dckProjPlot.addWidget(self.pltY)
        self.projCurveX=self.pltX.plot([],[])
        self.projCurveY=self.pltY.plot([],[])
        self.projCurveFitX=self.pltX.plot([],[], pen=(255,0,0))
        self.projCurveFitY=self.pltY.plot([],[], pen=(255,0,0))

        self.frame=frame
        self.oldCenter=getROICenter(rectRoi)
        self.rectRoi=rectRoi

        if area is None:
            area=da.DockArea()
        self.fitModel=None
        self.addToArea(area)
        self.model=bp.Gauss2dModel(*self.getDims())




    def addToArea(self, area):
        area.addDock(self.dck, 'right')
        area.addDock(self.dckProjPlot, 'bottom', self.dck)

    def getROIPars(self):
        roi=self.ellipseRoi
        pos=roi.pos()
        size=roi.size()
        sig1=size[0]
        sig2=size[1]
        theta=+roi.angle()/180*pi
        x0=pos[0]+(sig1*cos(theta) - sig2*sin(theta))/2
        y0=pos[1]+(sig1*sin(theta) + sig2*cos(theta))/2
        #y0= self.frame.shape[1]-y0
        return x0,y0, sig1, sig2, theta

    def setROIPars(self, x0, y0, sig1, sig2, theta):
        roi=self.ellipseRoi
        size=(sig1,sig2)
        theta=theta#*180/np.pi
        #y0=self.frame.shape[1]-y0
        pos=[x0-(sig1*cos(theta) - sin(theta)*sig2)/2, 
            y0-(sig1*sin(theta) + cos(theta)*sig2)/2]
        roi.setPos(pos)
        roi.setSize(size)
        roi.setAngle(theta*180/np.pi)
        print("size, pos, angle: {},{},{}".format(size,pos,theta))

    def guessFrameFromROI(self, bShow=False):
        x0,y0, sig1_0, sig2_0, theta0= self.getROIPars()
        self.model.setInitialPars([100, x0, y0, sig1_0, sig2_0, theta0])
        if bShow:
            pg.image(self.model.getFrame())

    def getDims(self):
        return self.frame.shape

    def updateFitProjections(self):
        if self.fitFrame is not None:
            self.projCurveFitX.setData(self.fitFrame.sum(axis=1))
            self.projCurveFitY.setData(self.fitFrame.sum(axis=0))

    def updateFrame(self, newFrame, newCenter=None):
        self.frame=newFrame
        self.subImg.setImage(newFrame)
        if newCenter is not None:
            movement=newCenter-self.oldCenter
            oldPos=np.array(self.ellipseRoi.pos())
            self.ellipseRoi.setPos(oldPos-movement)
            self.oldCenter=newCenter

        if self.bAutoFit:
            self.updateFit();
        if self.bUpdateProjections:
            self.projCurveX.setData(newFrame.sum(axis=1))
            self.projCurveY.setData(newFrame.sum(axis=0))
            self.updateFitProjections()
                

        #self.rectCenter

    def updateFit(self):
        x0,y0, sig1_0, sig2_0, theta0= self.getROIPars()
        print("initial: x0:{}, y0:{}, sig1:{}, sig2:{}, theta:{}".format(x0,y0,sig1_0/2,sig2_0/2,theta0))
        self.model=bp.Gauss2dModel(*self.getDims())
        if self.model.resObj is not None:
            height0=self.model.getP()[0]
        else:
            height0=self.frame.max()
        self.model.setInitialPars([height0, x0, y0, sig1_0, sig2_0, theta0])
        self.model.targetFrame=self.frame
        initialErr= self.model.getErr()
        if initialErr>14:
            fitP=self.model.fit_by_moments()
            self.model.setInitialPars(fitP)

        if self.bDoFullFit:
            self.model.fit(maxIters=30)
        #p,fitImg=bp.fit_2d_gaussian(frame.copy(), x0,y0, sig1_0/4, sig2_0/4, theta0, height0, ret_fit=True, maxIters=100)
            fitP=self.model.resObj.x
        print(fitP)
        self.fitFrame=self.model.getFramePar(fitP)
        self.lastFitP=fitP
        #pg.image(fitFrame)
        self.setROIPars(*fitP[1:])
        self.updateFitProjections()



# Parameters-----------------
paramDock=da.Dock("Parameters", size=(400,200))
fitParamDefs = [
    {'name': 'Fit Parameters', 'type': 'group', 'children': [
        {'name': 'pixel diameter', 'type': 'float', 'value':1, 'suffix':'um'},
        {'name': 'Max Iterations', 'type': 'int', 'value': 20},
        {'name': 'full fit', 'type': 'bool', 'value': False, 'tip': "Whether to do the full fit or just use moments"},
    ]},
    {'name': 'Fit Results', 'type': 'group', 'children': [
        {'name': 'major width', 'type': 'float', 'value': 1},
        {'name': 'minor width', 'type': 'float', 'value': 1},
        {'name': 'rotation angle', 'type': 'float', 'value': 0},
        {'name': 'peak height', 'type': 'float', 'value': 0},
        {'name': 'x0', 'type': 'float', 'value': 100},
        {'name': 'y0', 'type': 'float', 'value': 100},
    ]},
]

acquisitionParamDefs=[
    {'name': 'Acquisition Parameters', 'type': 'group', 'children': [
        {'name': 'Acquire Continuous', 'type': 'bool', 'value': False, 'tip': "Continually capture from the camera"},
        {'name': 'Capture Once', 'type': 'action'},
        {'name': 'Record Background', 'type': 'action'},
        {'name': 'Subtract BG', 'type': 'bool', 'value': False, 'tip': "Whether to subtract the saved background from the image"},
        {'name': 'Show BG', 'type': 'action', 'tip': "Show saved BG"},
        {'name': 'Averages', 'type': 'int', 'value': 1},
    ]}
]

actionParamDefs=[
    {'name': 'Actions', 'type': 'group', 'children': [
        {'name': 'Least-squares Fit', 'type': 'action'},
        {'name': 'Moments-based Fit', 'type': 'action'},
        {'name': 'Auto fit', 'type': 'bool', 'value': False, 'tip': "Whether to continually update the non-linear fit (slow)"},
        {'name': 'Update Projections', 'type': 'bool', 'value': True, 'tip': "Whether to update the projection plots"},
        {'name': 'Reset Crop', 'type': 'action'},
        
    ]},
]

## Create tree of Parameter objects
par = Parameter.create(name='params', type='group', children=fitParamDefs+acquisitionParamDefs+actionParamDefs)
pTree = ParameterTree()
pTree.setParameters(par, showTop=False)
pTree.setWindowTitle('pyqtgraph example: Parameter Tree')
paramDock.addWidget(pTree)
#par.child('Actions', 'Fit Once').connect(crp.updateFit())

bCaptureContinuous=False
def paramChanged(param, changes):
    for param, changeType, newVal in changes:
        print("Handling name: {}, changeType: {}, newVal: {}".format(param.name, changeType, newVal))
        name=param.name()
        if changeType=='activated':
            if name=="Capture Once":
                capture()
            if name=="Least-squares Fit":
                crp.updateFit()
            if name=="Record Background":
                main.setBG()
            if name=="Reset Crop":
                main.resetCrop()
            if name=="Show BG":
                pg.image(main.bg)
        elif name=='Acquire Continuous':
            global bCaptureContinuous
            bCaptureContinuous=newVal 
        elif name=='full fit':
            crp.bDoFullFit=newVal 
        elif name=='Subtract BG':
            main.bSubtractBG=newVal 
        elif name=='Update Projections':
            main.bUpdateProjections=newVal 
        elif name=='Auto fit':
            crp.bAutoFit=newVal 
        elif name=='Averages':
            cam.Naves=newVal 
        else:
            print("Somehow missed handling '{}'".format(name))
par.sigTreeStateChanged.connect(paramChanged)
#win.show()

## lock the aspect ratio so pixels are always square
#view.setAspectLocked(True)

## Create image item
#img = pg.ImageItem(border='w')
#view.addItem(img)
#img=imageWidget.getImageItem()

## Set initial view bounds
#view.setRange(QtCore.QRectF(0, 0, 640, 480))


updateTime = ptime.time()
fps = 0
frame=None
cam=DummyCam()

main=MainArea(frame=cam.get_frame_bw(), area=area)
crp=Cropped(main.subFrame, main.rectRoi, area )        
area.addDock(paramDock, 'bottom', main.dck)

def capture():
    global i, updateTime, fps, bCaptureContinuous
    #camimg=cam.get_image()
    #arr=pygame.surfarray.pixels3d(camimg)
    ## Display the data
    frame=cam.get_frame_bw()
    #img.setImage(arr[:,:,:])
    #arr=np.random.normal(size=(640,480))**2
    main.updateFrame(frame)
    crp.updateFrame(main.subFrame)
    #img.setImage(frame)
    #print("maj: {}, min: {}".format(outD['major_axis'], outD['minor_axis']))
    #img2.setImage(outD['frame'])
    #updateFit()


    
    if bCaptureContinuous:
        QtCore.QTimer.singleShot(1, capture)
        now = ptime.time()
        fps2 = 1.0 / (now-updateTime)
        updateTime = now
        fps = fps * 0.9 + fps2 * 0.1
    
    print("%0.1f fps" % fps)
   
def cropRegionMoved(roi):
    #pdb.set_trace()
    newCenter=getROICenter(roi)
    crp.updateFrame(main.subFrame, newCenter)

main.rectRoi.sigRegionChanged.connect(cropRegionMoved)
timer=QtCore.QTimer()#.singleShot(1, updateData)
#timer.timeout.connect(capture)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    #timer.start(30000)
    #updateData()
    #data = np.random.normal(size=(640, 600), loc=1024, scale=64).astype(np.uint16)
    capture()
    win.show()
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pass;
        #QtGui.QApplication.instance().exec_()
