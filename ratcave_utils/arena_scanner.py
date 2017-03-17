__author__ = 'nickdg'



import click
import motive
import numpy as np
import pyglet
import pyglet.gl as gl
import ratcave as rc
from os import path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wavefront_reader import WavefrontWriter

from . import cli

from _transformations import rotation_matrix
from ratcave_utils.utils import hardware, pointcloud

np.set_printoptions(precision=3, suppress=True)


class GridScanWindow(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        """
        Returns Window with everything needed for doing arena scanning.  Will automatically close when all camera movement
        has completed.
        """
        super(GridScanWindow, self).__init__(*args, **kwargs)

        wavefront_reader = rc.WavefrontReader(rc.resources.obj_primitives)
        self.mesh = wavefront_reader.get_mesh('Grid', position=[0., 0., -1.], scale=1.5)
        self.mesh.drawmode = rc.POINTS
        self.mesh.gl_states = (gl.GL_POINT_SMOOTH,)
        self.mesh.uniforms['diffuse'] = [1., 1., 1.]  # Make white
        self.mesh.uniforms['flat_shading'] = True

        self.scene = rc.Scene([self.mesh], bgColor=(0, 0, 0))
        self.scene.camera.ortho_mode = True

        self.shader = rc.Shader.from_file(*rc.resources.genShader)

        dist = .06
        self.cam_positions = ((dist * np.sin(ang), dist * np.cos(ang), 0) for ang in np.linspace(0, 2*np.pi, 40)[:-1])

        self.marker_pos = []
        pyglet.clock.schedule(self.detect_projection_point)
        pyglet.clock.schedule(self.move_camera)

    def move_camera(self, dt):
        """Randomly moves the mesh center to somewhere between xlim and ylim"""
        try:
            self.scene.camera.position.xyz = next(self.cam_positions)
        except StopIteration:
            print("End of Camera Position list reached. Closing window...")
            pyglet.clock.unschedule(self.detect_projection_point)
            self.close()

    def on_draw(self):
        """Render the scene!"""
        with self.shader:
            gl.glPointSize(10.)
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
@click.option('--nsides', help='Number of Arena Sides.  If not given, will be estimated from data.', type=int, default=0)
@click.option('--screen', help='Screen number to display on', default=1, type=int)
def scan_arena(motive_filename, output_filename, body, nomeancenter, nsides, screen):
    """Runs Arena Scanning algorithm."""

    output_filename = output_filename + '.obj' if not path.splitext(output_filename)[1] else output_filename
    assert path.splitext(output_filename)[1] == '.obj', "Output arena filename must be a Wavefront (.obj) file"

    # Load Motive Project File
    motive_filename = motive_filename.encode()
    motive.initialize()
    motive.load_project(motive_filename)
    hardware.motive_camera_vislight_configure()
    motive.update()

    # Get Arena's Rigid Body
    rigid_bodies = motive.get_rigid_bodies()
    assert body in rigid_bodies, "RigidBody {} not found in project file.  Available body names: {}".format(body, list(rigid_bodies.keys()))
    assert len(rigid_bodies[body].markers) > 5, "At least 6 markers in the arena's rigid body is required. Only {} found".format(len(rigid_bodies[body].markers))

    # Scan points
    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[screen]
    window = GridScanWindow(screen=screen, fullscreen=True)
    pyglet.app.run()
    points = np.array(window.marker_pos)
    assert(len(points) > 100), "Only {} points detected.  Tracker is not detecting enough points to model.  Is the projector turned on?".format(len(points))
    assert points.ndim == 2


    # Plot preview of data collected
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points.T)
    plt.show()

    # Get vertex positions and normal directions from the collected data.
    vertices, normals = pointcloud.meshify_arena(points, n_surfaces=nsides)


    # vertices = {wall: pointcloud.fan_triangulate(pointcloud.reorder_vertices(verts)) for wall, verts in vertices.items()}  # Triangulate
    vertices = np.array([pointcloud.fan_triangulate(pointcloud.reorder_vertices(verts)) for verts in vertices])  # Triangulate


    normals = normals.reshape(-1, 1, 3).repeat(vertices.shape[1], axis=1).reshape(-1, 3)
    vertices = vertices.reshape(-1, 3)

    # me
    if not nomeancenter:
        vertmean = np.mean(vertices, axis=0)
        # vertmean = np.array([np.mean(np.unique(verts)) for verts in vertices.T])  # to avoid counting the same vertices twice.
        vertices -= vertmean
        points -= vertmean
        rigid_bodies[body].location = vertmean



    # Write wavefront .obj file to app data directory and user-specified directory for importing into Blender.
    writer = WavefrontWriter.from_arrays(body, vertices, normals)
    with open(output_filename, 'w') as f:
        writer.dump(output_filename)
        f.write('')

    # Show resulting plot with points and model in same place.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*points[::12, :].T)
    ax.scatter(*vertices.T, c='r')
    plt.show()


    for el in range(3):
        rigid_bodies[body].reset_orientation()
        motive.update()
    motive.save_project(motive_filename)




if __name__ == '__main__':
    scan_arena()