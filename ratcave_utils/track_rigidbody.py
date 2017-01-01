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
        self.mesh.rotation = self.mesh.rotation.to_quaternion()
        self.scene = rc.Scene(meshes=[self.mesh])

        self.label = pyglet.text.Label()

    def on_draw(self):
        self.scene.draw()
        self.label.draw()


@cli.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.option('--body', help='Name of rigidBody to track')
def trackrotation(motive_filename, body):

    motive.initialize()
    motive_filename = motive_filename.encode()
    motive.load_project(motive_filename)

    body = motive.get_rigid_bodies()[body]
    for el in range(3):
        body.reset_orientation()
        motive.update()

    window = RotationWindow()

    def update_body(dt, window, body):
        motive.update()
        window.mesh.rotation.xyzw = body.rotation_quats

        fmt_str = "loc: {:.1f}, {:.1f}, {:.1f}\t rot: {:.2f}, {:.2f}, {:.2f}, {:.2f}"
        window.label.text = fmt_str.format(*(body.location + body.rotation_quats))

    pyglet.clock.schedule(update_body, window, body)

    pyglet.app.run()


