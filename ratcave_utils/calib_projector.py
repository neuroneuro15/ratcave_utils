__author__ = 'nickdg'

import sys
import random
import pickle
from os import path

import click
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pyglet
import pyglet.gl as gl

import motive
import ratcave as rc

from . import cli
from ratcave_utils.utils import hardware
import _transformations as trans

np.set_printoptions(precision=3, suppress=True)


class PointScanWindow(pyglet.window.Window):

    def __init__(self, max_points=1000, *args, **kwargs):
        """
        Returns Window with everything needed for doing projector calibration.

        Keyword Args:
            -max_points (int): total number of data points to collect before exiting.

        """
        super(PointScanWindow, self).__init__(*args, **kwargs)
        gl.glPointSize(20)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        self.curr_pos = 0, 0
        
        self.max_points = max_points
        self.screen_pos = []
        self.marker_pos = []
        pyglet.clock.schedule(self.detect_projection_point)
        pyglet.clock.schedule(self._close_if_max_points_reached)

    def _close_if_max_points_reached(self, dt):
        if len(self.screen_pos) >= self.max_points:
            self.close()

    def on_draw(self):
        """Move the mesh, then draw it!"""
        self.clear()
        self.curr_pos = random.randint(0, self.width), random.randint(0, self.height)
        pyglet.graphics.draw(1, gl.GL_POINTS, ('v2i', self.curr_pos), ('c3B', (255, 255, 255)))

    def detect_projection_point(self, dt):
        """Use Motive to detect the projected mesh in 3D space"""
        motive.flush_camera_queues()
        for el in range(2):
            motive.update()

        markers = motive.get_unident_markers()
        markers = list(markers)# if marker[1] < 0.50 and marker[1] > 0.08]
        if len(markers) == 1:
            click.echo(markers)
            x, y = self.curr_pos
            self.screen_pos.append([x/self.width - 0.5, (y - self.height/2) / self.width])
            self.marker_pos.append(markers[0])



def calibrate(img_points, obj_points):
    """
    Returns position and rotation arrays by using OpenCV's camera calibration function on image calibration data.

    Args:
        -img_points (Nx2 NumPy Array): the location (-.5 - .5) of the center of the point that was projected on the
            projected image.
        -obj_points (Nx3 NumPy Array): the location (x,y,z) in real space where the projected point was measured.
            Note: Y-axis is currently hardcoded to represent the 'up' direction.

    Returns:
        -posVec (NumPy Array): The X,Y,Z position of the projector
        -rotVec (NumPy Array): The Euler3D rotation of the projector (in degrees).
    """
    img_points, obj_points = np.array(img_points, dtype=np.float32), np.array(obj_points, dtype=np.float32)
    assert img_points.ndim == 2
    assert obj_points.ndim == 2
    img_points *= -1

    _, cam_mat, _, rotVec, posVec = cv2.calibrateCamera([obj_points], [img_points], (1, 1),#,  # Currently a false window size. # TODO: Get cv2.calibrateCamera to return correct intrinsic parameters.
                                        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | # cv2.CALIB_FIX_ASPECT_RATIO | # Assumes equal height/width aspect ratio
                                              cv2.CALIB_ZERO_TANGENT_DIST |  cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                                              cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
                                                        )
    # Change order of coordinates from cv2's camera-centered coordinates to Optitrack y-up coords.
    pV, rV = posVec[0], rotVec[0]

    # Return the position array and rotation matrix for the camera.
    position = np.dot(pV.T, cv2.Rodrigues(rV)[0]).flatten()  # Invert the position by the rotation to be back in world coordinates
    camera_matrix = cv2.Rodrigues(rV)[0]

    # Build Model Matrix from openCV output: convert to view matrix, then invert to get model matrix.
    model_matrix = np.identity(4)
    model_matrix[:3, -1] = -position
    model_matrix[:3, :3] = np.linalg.inv(camera_matrix)

    return model_matrix


def plot_estimate(obj_points, position, rotation_matrix):
    """
    Make a 3D plot of the data and the projector position and direction estimate, just to verify that the estimate
    makes sense.
    """

    obj_points = np.array(obj_points)
    assert obj_points.ndim == 2

    cam_dir = np.dot([0, 0, -1], rotation_matrix) # Rotate from -Z vector (default OpenGL camera direction)
    rot_vec = np.vstack((position, position+cam_dir))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*obj_points.T)
    ax.plot(*rot_vec.T)
    plt.show()



def plot2d(img_points, obj_points):
    """Verify that the image data and marker data is not random by plotting xy relationship between them."""
    img_points = np.array(img_points)
    obj_points = np.array(obj_points)
    assert img_points.ndim == 2 and obj_points.ndim == 2
    fig, axes = plt.subplots(ncols=2)
    for idx in range(2):
        for obj_coord, label in zip(obj_points.T, 'xyz'):
            axes[idx].plot(img_points[:,idx], obj_coord, 'o', label=label)
        axes[idx].legend()

    plt.show()


@cli.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.argument('projector_filename', type=click.Path())
@click.option('--points', default=100, help="Number of data points to collect before estimating position")
@click.option('--fps', default=15, help="Frame rate to update window at.")
@click.option('--screen', default=1, help='Screen number to display on.')
def calib_projector(motive_filename, projector_filename, points, fps, screen):

    # Verify inputs
    if not path.splitext(projector_filename)[1]:
        projector_filename += '.pickle'
    projector_filename = projector_filename.encode()

    if sys.version_info.major == 2:
        motive_filename = motive_filename.encode()

    # Collect Data
    motive.initialize()
    motive.load_project(motive_filename)


    hardware.motive_camera_vislight_configure()

    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[screen]
    pyglet.clock.set_fps_limit(fps)
    window = PointScanWindow(screen=screen, fullscreen=True, max_points=points)
    pyglet.app.run()

    # Run Calibration Algorithm and Plot results
    model_matrix = calibrate(window.screen_pos, window.marker_pos)
    click.echo(model_matrix)
    plot2d(window.screen_pos, window.marker_pos)
    plot_estimate(obj_points=window.marker_pos, position=model_matrix[:3, -1], rotation_matrix=np.linalg.inv(model_matrix[:3, :3]))

    # Create RatCAVE Camera for use in the project and save it in a pickle file.
    camera = rc.Camera()
    camera.position.xyz = model_matrix[:3, -1]
    camera.rotation = camera.rotation.from_matrix(model_matrix)
    camera.update()
    camera.projection.fov_y = .0338 * window.height + 4.47  # Calculated just for our projector.
    camera.projection.update()
    click.echo(camera.position)
    click.echo(camera.rotation)

    with open(projector_filename, 'wb') as f:
        pickle.dump(camera, f)

if __name__ == '__main__':
    calib_projector()
