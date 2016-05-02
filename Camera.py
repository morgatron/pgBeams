from traits.api import HasTraits, Int, Str, Tuple, Array, Range
from traitsui.api import View, Label
import numpy as np

class CameraError(Exception):
    def __init__(self, msg, cam):
        self.msg = msg
        self.camera_number = cam

    def __str__(self):
        return '{0} on camera {1}'.format(self.msg, self.camera_number)


class Camera(HasTraits):
    Naves=1
    camera_number = Int(-1)
    id_string = Str()
    resolution = Tuple(Int(), Int())
    roi = Tuple(Int(), Int(), Int(), Int())
    frame_rate = Range(1, 500, 30)
    frame = Array()

    # Default configuration panel
    view = View(Label('No settings to configure'))

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
        return False  # don't suppress exceptions

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def query_frame(self):
        raise NotImplementedError()

    def get_frame_bw(self):
        #Could put some saturation checking in here
        self.query_frame()
        frameTot=self.frame.copy()
        for k in range(self.Naves-1):
            self.query_frame()
            frameTot+=self.frame
        if self.Naves>1:
            frameTot/=self.Naves

        frame=self.frame
        Nsat=(frame>254).sum()
        if Nsat:
            print("Beware: at least {} saturated pixels!!".format(Nsat))
        if frame.ndim>2:
            frame= np.array(
                0.2989 * frame[..., 0]
                + 0.5870 * frame[..., 1]
                + 0.1140 * frame[..., 2])
        return frame

    def find_resolutions(self):
        '''
        Returns a list of resolution tuples that this camera supports.
        '''
        # Default: return the camera's own default resolution
        return [self.resolution]

    def configure(self):
        """Opens a dialog to set the camera's parameters."""
        pass
