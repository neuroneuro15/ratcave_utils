import time
import motive
import pyglet
from pyglet.window import key
import click
import ratcave as rc
from . import cli
import numpy as np
import pickle

class RotationWindow(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        super(RotationWindow, self).__init__(*args, **kwargs)

        pyglet.clock.schedule(lambda dt: None)

        reader = rc.WavefrontReader(rc.resources.obj_primitives)
        self.mesh = reader.get_mesh('Monkey', scale=.03, position=(0, 0, 0))
        self.mesh.rotation = self.mesh.rotation.to_quaternion()
        self.scene = rc.Scene(meshes=[self.mesh], bgColor=(1., 0., 0.))

        self.label = pyglet.text.Label()

    def on_draw(self):
        with rc.resources.genShader:
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
    window.mesh.position.z = -1.5

    def update_body(dt, window, body):
        motive.update()
        window.mesh.rotation.xyzw = body.rotation_quats

        fmt_str = "loc: {:.1f}, {:.1f}, {:.1f}\t rot: {:.2f}, {:.2f}, {:.2f}, {:.2f}"
        window.label.text = fmt_str.format(*(body.location + body.rotation_quats))

    pyglet.clock.schedule(update_body, window, body)

    pyglet.app.run()


@cli.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.argument('projector_filename', type=click.Path(exists=True))
@click.option('--body', help='Name of rigidBody to track')
@click.option('--screen', help='Screen Number to display on', type=int, default=1)
def trackposition(motive_filename, projector_filename, body, screen):

    motive.initialize()
    motive_filename = motive_filename.encode()
    motive.load_project(motive_filename)

    body = motive.get_rigid_bodies()[body]
    for el in range(3):
        body.reset_orientation()
        motive.update()


    # Load projector's Camera object, created from the calib_projector ratcave_utils CLI tool.



    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[screen]
    window = RotationWindow(fullscreen=True, screen=screen)

    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    camera = rc.Camera.from_pickle(projector_filename.encode())
    window.scene.camera = camera


    @window.event
    def on_draw():
        with rc.resources.genShader:
            window.scene.draw()
        window.label.draw()
        # print(window.scene.camera.rotation)


    def update_body(dt, window, body):
        motive.update()
        window.mesh.rotation.xyzw = body.rotation_quats
        window.mesh.position.xyz = body.location
        # window.scene.camera.rotation.x += 15 * dt
        fmt_str = "loc: {:.1f}, {:.1f}, {:.1f}\t rot: {:.2f}, {:.2f}, {:.2f}, {:.2f}\n{fov_y:.2f}\t{aspect:.2f}"
        window.label.text = fmt_str.format(*(body.location + body.rotation_quats),
                                       fov_y=window.scene.camera.projection.fov_y,
                                           aspect=window.scene.camera.projection.aspect)

        # print(window.scene.camera.projection.projection_matrix)
        print(window.scene.camera.uniforms['projection_matrix'])

    def update_fov(dt):

        camera.projection.aspect = window.width / float(window.height)
        camera.projection.update()

        speed = 10.  # How fast to change values on keyboard hold.
        if keys[key.UP]:
            camera.projection.fov_y += speed * dt
            camera.projection.update()
        if keys[key.DOWN]:
            camera.projection.fov_y -= speed * dt
            camera.projection.update()
        if keys[key.LEFT]:
            camera.projection.aspect += speed * dt
            camera.projection.update()
        if keys[key.RIGHT]:
            camera.projection.aspect -= speed * dt
            camera.projection.update()


    pyglet.clock.schedule(update_fov)






    pyglet.clock.schedule(update_body, window, body)

    pyglet.app.run()


