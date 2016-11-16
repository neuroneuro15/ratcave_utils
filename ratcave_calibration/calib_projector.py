__author__ = 'nickdg'

import random

import click
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import motive
import numpy as np
import pyglet
import ratcave as rc

from ratcave_calibration.utils import hardware

np.set_printoptions(precision=3, suppress=True)




class PointScanWindow(pyglet.window.Window):

    def __init__(self, max_points=1000, *args, **kwargs):
        """
        Returns Window with everything needed for doing projector calibration.

        Keyword Args:
            -max_points (int): total number of data points to collect before exiting.

        """
        super(PointScanWindow, self).__init__(*args, **kwargs)

        wavefront_reader = rc.WavefrontReader(rc.resources.obj_primitives)
        self.mesh = wavefront_reader.get_mesh('Sphere', position=[0., 0., -1.], scale=.01)
        self.mesh.uniforms['diffuse'] = [1., 1., 1.]  # Make white
        self.mesh.uniforms['flat_shading'] = True

        self.scene = rc.Scene([self.mesh], bgColor=(0, 0, 0))
        self.scene.camera.ortho_mode = True

        self.max_points = max_points
        self.screen_pos = []
        self.marker_pos = []
        pyglet.clock.schedule(self.detect_projection_point)
        pyglet.clock.schedule(self._close_if_max_points_reached)

    def _close_if_max_points_reached(self, dt):
        if len(self.screen_pos) >= self.max_points:
            self.close()

    def randomly_move_point(self, xlim=(-.9, .9), ylim=(-.5, .5)):
        """Randomly moves the mesh center to somewhere between xlim and ylim"""
        for attr, lims in zip('xy', (xlim, ylim)):
            limrange = max(lims) - min(lims)
            newpos = limrange * random.random() - (limrange / 2)
            setattr(self.mesh, attr, newpos)

    def on_draw(self):
        """Move the mesh, then draw it!"""
        self.randomly_move_point()
        self.scene.draw()

    def detect_projection_point(self, dt):
        """Use Motive to detect the projected mesh in 3D space"""
        motive.flush_camera_queues()
        for el in range(2):
            motive.update()

        markers = motive.get_unident_markers()
        markers = [marker for marker in markers if marker[1] < 0.50 and marker[1] > 0.08]
        if len(markers) == 1:
            click.echo(markers)
            self.screen_pos.append([self.mesh.x, self.mesh.y])
            self.marker_pos.append(markers[0])





@click.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.option('--debug', default=False, help='Display window only (no logging)')
@click.option('--points', default=100, help="Number of data points to collect before estimating position")
@click.option('--fps', default=15, help="Frame rate to update window at.")
def run(motive_filename, debug, points, fps):


    motive_filename = motive_filename.encode()

    motive.load_project(motive_filename)
    hardware.motive_camera_vislight_configure()

    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[1]


    pyglet.clock.set_fps_limit(fps)
    window = PointScanWindow(screen=screen, fullscreen=True, max_points=points)


    pyglet.app.run()
    pos, rotmat = calibrate(window.screen_pos, window.marker_pos)
    click.echo(pos)
    click.echo(rotmat)
    plot2d(window.screen_pos, window.marker_pos)

    plot_estimate(obj_points=window.marker_pos, position=pos, rotation_matrix=rotmat)




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

    _, cam_mat, _, rotVec, posVec = cv2.calibrateCamera([obj_points], [img_points], (1,1),  # Currently a false window size. # TODO: Get cv2.calibrateCamera to return correct intrinsic parameters.
                                        flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | # Assumes equal height/width aspect ratio
                                              cv2.CALIB_ZERO_TANGENT_DIST |  cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                                              cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)

    # Change order of coordinates from cv2's camera-centered coordinates to Optitrack y-up coords.
    pV, rV = posVec[0], rotVec[0]

    # Return the position array and rotation matrix for the camera.
    position = -np.dot(pV.T, cv2.Rodrigues(rV)[0]).flatten()  # Invert the position by the rotation to be back in world coordinates
    rotation_matrix = cv2.Rodrigues(rV)[0]

    return position, rotation_matrix


def plot_estimate(obj_points, position, rotation_matrix):
    """Make a 3D plot of the data and the projector position and direction estimate, just to verify that the estimate
    makes sense."""

    obj_points = np.array(obj_points)
    assert obj_points.ndim == 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*obj_points.T)

    plt.show()

    # # Plot and Check that vector position and direction is generally correct, and give tips if not.
    # cam_dir = np.dot([0, 0, -1], rotation_matrix) # Rotate from -Z vector (default OpenGL camera direction)
    # rot_vec = np.vstack((position, position+cam_dir))
    #
    #
    #
    #
    # ax.scatter(obj)
    # ax = plot_3d(pointPos, square_axis=True)
    # plot_3d(rot_vec, ax=ax, color='r', line=True)
    # plot_3d(np.array([position]), ax=ax, color='g', show=True)


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



if __name__ == '__main__':
    run()
