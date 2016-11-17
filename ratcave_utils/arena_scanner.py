__author__ = 'nickdg'



import click
import motive
import numpy as np
import pyglet
import ratcave as rc
from os import path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import cli

from _transformations import rotation_matrix
from ratcave_utils.utils import orienting, hardware, pointcloud

np.set_printoptions(precision=3, suppress=True)


class GridScanWindow(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        """
        Returns Window with everything needed for doing arena scanning.  Will automatically close when all camera movement
        has completed.
        """
        super(GridScanWindow, self).__init__(*args, **kwargs)

        wavefront_reader = rc.WavefrontReader(rc.resources.obj_primitives)
        self.mesh = wavefront_reader.get_mesh('Grid', position=[0., 0., -1.], scale=1.5, point_size=12, drawstyle='point')
        self.mesh.uniforms['diffuse'] = [1., 1., 1.]  # Make white
        self.mesh.uniforms['flat_shading'] = True

        self.scene = rc.Scene([self.mesh], bgColor=(0, 0, 0))
        self.scene.camera.ortho_mode = True

        dist = .06
        self.cam_positions = ((dist * np.sin(ang), dist * np.cos(ang), -1) for ang in np.linspace(0, 2*np.pi, 40)[:-1])

        self.marker_pos = []
        pyglet.clock.schedule(self.detect_projection_point)
        pyglet.clock.schedule(self.move_camera)

    def move_camera(self, dt):
        """Randomly moves the mesh center to somewhere between xlim and ylim"""
        try:
            self.scene.camera.position = next(self.cam_positions)
        except StopIteration:
            print("End of Camera Position list reached. Closing window...")
            pyglet.clock.unschedule(self.detect_projection_point)
            self.close()

    def on_draw(self):
        """Render the scene!"""
        self.scene.draw()

    def detect_projection_point(self, dt):
        """Use Motive to detect the projected mesh in 3D space"""
        motive.flush_camera_queues()
        for el in range(2):
            motive.update()

        markers = motive.get_unident_markers()
        markers = [marker for marker in markers if 0.08 < marker[1] < 0.50]

        click.echo("{} markers detected.".format(len(markers)))
        self.marker_pos.extend(markers)


@cli.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.argument('output_filename', type=click.Path())
@click.option('--body', help='Name of arena rigidbody to track', default='arena')
@click.option('--nomeancenter', help='Flag: Skips mean-centering of arena.', type=bool, default=False)
@click.option('--nopca', help='Flag: skips PCA-based arena marker rotation (used for aligning IR markers to image points)', type=bool, default=False)
@click.option('--nsides', help='Number of Arena Sides.  If not given, will be estimated from data.', type=int, default=0)
def scan_arena(motive_filename, output_filename, body, nomeancenter, nopca, nsides):
    """Runs Arena Scanning algorithm."""

    output_filename = output_filename + '.obj' if not path.splitext(output_filename)[1] else output_filename
    assert path.splitext(output_filename)[1] == '.obj', "Output arena filename must be a Wavefront (.obj) file"

    # Load Motive Project File
    motive_filename = motive_filename.encode()
    motive.load_project(motive_filename)
    hardware.motive_camera_vislight_configure()
    motive.update()

    # Get Arena's Rigid Body
    rigid_bodies = motive.get_rigid_bodies()
    assert body in rigid_bodies, "RigidBody {} not found in project file.  Available body names: {}".format(body, list(rigid_bodies.keys()))
    assert len(rigid_bodies[body].markers) > 5, "At least 6 markers in the arena's rigid body is required. Only {} found".format(len(rigid_bodies[body].markers))


    # TODO: Fix bug that requires scanning be done in original orientation (doesn't affect later recreation, luckily.)
    for attempt in range(3):  # Sometimes it doesn't work on the first try, for some reason.
        rigid_bodies[body].reset_orientation()
        motive.update()
        if sum(np.abs(rigid_bodies[body].rotation)) < 1.:
            break
    else:
        raise ValueError("Rigid Body Orientation not Resetting to 0,0,0 after 3 attempts.  This happens sometimes (bug), please just run the script again.")

    # Scan points
    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[1]
    window = GridScanWindow(screen=screen, fullscreen=True)
    pyglet.app.run()
    points = np.array(window.marker_pos)
    assert(len(points) > 100), "Only {} points detected.  Tracker is not detecting enough points to model.  Is the projector turned on?".format(len(points))
    assert points.ndim == 2

    # Rotate all points to be mean-centered and aligned to Optitrack Markers direction or largest variance.
    markers = np.array(rigid_bodies[body].point_cloud_markers)
    if not nomeancenter:
        points -= np.mean(markers, axis=0)
    if not nopca:
        points = np.dot(points, rotation_matrix(np.radians(orienting.rotate_to_var(markers)), [0, 1, 0])[:3, :3])

    # Plot preview of data collected
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points.T)
    plt.show()

    # Get vertex positions and normal directions from the collected data.
    vertices, normals = pointcloud.meshify(points, n_surfaces=nsides)
    vertices = {wall: pointcloud.fan_triangulate(pointcloud.reorder_vertices(verts)) for wall, verts in vertices.items()}  # Triangulate

    # Write wavefront .obj file to app data directory and user-specified directory for importing into Blender.
    wave_str = pointcloud.to_wavefront(body, vertices, normals)
    with open(output_filename, 'wb') as wavfile:
        wavfile.write(wave_str)

    # Show resulting plot with points and model in same place.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points[::12, :].T)
    for idx, verts in vertices.items():
        ax.plot(*np.vstack((verts, verts[0, :])).T)
    plt.show()




if __name__ == '__main__':
    scan_arena()