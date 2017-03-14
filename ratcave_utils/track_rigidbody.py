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
        self.mesh = reader.get_mesh('Monkey', scale=.05, position=(0, 0, 0))
        self.mesh.rotation = self.mesh.rotation.to_quaternion()
        self.scene = rc.Scene(meshes=[self.mesh], bgColor=(0., 0., 0.))

        self.fbo = rc.FBO(rc.Texture(width=4096, height=4096))
        self.quad = rc.gen_fullscreen_quad()
        self.quad.texture = self.fbo.texture

        self.label = pyglet.text.Label()

        self.shader3d = rc.Shader.from_file(*rc.resources.genShader)
        self.shaderAA = rc.Shader.from_file(*rc.resources.deferredShader)

    def on_draw(self):
        with self.shader3d, self.fbo:
            self.scene.draw()
        with self.shaderAA:
            self.quad.draw()
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
    window.scene.light.position.xyz = camera.position.xyz

    shader = rc.Shader.from_file(*rc.resources.genShader)


    @window.event
    def on_draw():
        with shader:
            window.scene.draw()
        window.label.draw()


    def update_body(dt, window, body):
        motive.update()
        window.mesh.rotation.xyzw = body.rotation_quats
        window.mesh.position.xyz = body.location
        # window.scene.camera.rotation.x += 15 * dt
        fmt_str = "loc: {:.1f}, {:.1f}, {:.1f}\t rot: {:.2f}, {:.2f}, {:.2f}, {:.2f}\nfov_y: {fov_y:.2f}\taspect: {aspect:.2f}\nfps: {fps:.1f}"
        window.label.text = fmt_str.format(*(body.location + body.rotation_quats),
                                       fov_y=window.scene.camera.projection.fov_y,
                                           aspect=window.scene.camera.projection.aspect,
                                           fps=1./(dt + .00000001))


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


