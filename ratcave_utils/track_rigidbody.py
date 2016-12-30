import time
import motive
import pyglet
import click
import ratcave as rc
from . import cli
import numpy as np

class RotationWindow(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        super(RotationWindow, self).__init__(*args, **kwargs)

        pyglet.clock.schedule(lambda dt: None)

        reader = rc.WavefrontReader(rc.resources.obj_primitives)
        self.mesh = reader.get_mesh('Monkey', scale=.3, position=(0, 0, -1.5))
        self.scene = rc.Scene(meshes=[self.mesh])

        self.label = pyglet.text.Label()

    def on_draw(self):
        self.scene.draw()
        self.label.draw()

@cli.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.option('--body', help='Name of rigidBody to track')
def trackbody(motive_filename, body):

    # motive.initialize()
    #
    # motive_filename = motive_filename.encode()
    # motive.load_project(motive_filename)
    #
    # body = motive.get_rigid_bodies()[body]
    # for el in range(3):
    #     body.reset_orientation()
    #     motive.update()

    window = RotationWindow()

    window.mesh._old_rotation = np.zeros(3, dtype=float)
    window.mesh._old_roty = 0.
    window.mesh.change_rot = True
    window.scene.camera.old_time = 0.
    def update_body(dt, window, body):
        # motive.update()
        # # fmt_str = "loc: {:.1f}, {:.1f}, {:.1f}\t rot: {:.1f}, {:.1f}, {:.1f}, {:.1f}"
        # # window.label.text = fmt_str.format(*(body.location + body.rotation_quats))
        # print(type(body.rotation_quats))
        # if len(body.rotation_quats) == 4:
        #     window.mesh.rotation_quaternions = body.rotation_quats
        #     window.mesh.update()
        #     print(window.mesh.rotation)
        #     print(window.mesh.rotation_quaternions)
        #     print(window.mesh.model_matrix)
        #     print('')
        # else:
        #     print(body.rotation_quats)
        # window.mesh.change_rot = not window.mesh.change_rot
        # if window.mesh.change_rot:
        #     print('changing rotation in script...')
            # window.mesh.rot[1] += 36. * dt
            # window.mesh.rot_y += 36. * dt
            # window.mesh.rot.y += 36. * dt
            # window.mesh._old_roty += 36. * dt
            # window.mesh.rot[:] = (0., window.mesh._old_roty, 0.)

        window.mesh._old_rotation += (0., 36. * dt, 0.)
        # window.mesh.rotation = window.mesh._old_rotation
        window.mesh.rot.xyz = window.mesh._old_rotation
        window.scene.camera.old_time += dt
        window.scene.camera.rot.y = 20. * np.sin(window.scene.camera.old_time)

        print(window.mesh.rot)

    pyglet.clock.schedule(update_body, window, body)

    pyglet.app.run()


