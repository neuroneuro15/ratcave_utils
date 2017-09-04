import motive
import pyglet
from pyglet.window import key
import click
import ratcave as rc
from . import cli
import serial

class LatencyDisplayApp(pyglet.window.Window):

    def __init__(self, projector, serial_device, *args, **kwargs):
        super(LatencyDisplayApp, self).__init__(*args, **kwargs)

        self.device = serial_device

        pyglet.clock.schedule(lambda dt: None)

        reader = rc.WavefrontReader(rc.resources.obj_primitives)
        self.dot = reader.get_mesh('Sphere', scale=.02, position=(0, 0, 0))
        self.dot.uniforms['diffuse'] = 1., 0., 0.
        self.dot.uniforms['flat_shading'] = True
        self.scene = rc.Scene(meshes=[self.dot], bgColor=(0., 0., 0.))
        self.scene.camera = projector
        self.scene.camera.projection.aspect = 1.7777
        self.shader3d = rc.Shader.from_file(*rc.resources.genShader)

        self.latency_display = pyglet.text.Label(text='Waiting for Stimulus Switch..', x=0, y=0, font_size=36)
        pyglet.clock.schedule(self.update_latency_display)

    def on_draw(self):
        with self.shader3d:
            self.scene.draw()
        self.latency_display.draw()

    def update_latency_display(self, dt):
        if self.device.in_waiting:
            self.latency_display.text = "Latency: {}".format(self.device.readline())

    def on_key_release(self, sym, mod):
        if sym == key.LEFT:
            self.device.write(b'A')
        elif sym == key.RIGHT:
            self.device.write(b'B')

    # def on_resize(self, width, height):
    #     self.scene.camera.projection.match_aspect_to_viewport()



@cli.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.argument('projector_filename', type=click.Path(exists=True))
@click.option('--port', help='Serial Port Name that Latency Device is Connected to.', default='COM7')
@click.option('--body', help='Name of rigidBody to track')
@click.option('--screen', help='Screen Number to display on', type=int, default=1)
def latency_test(motive_filename, projector_filename, port, body, screen):

    with serial.Serial(port=port, timeout=.5) as device:

        device.write('A')

        motive.initialize()
        motive.load_project(motive_filename.encode())
        motive.update()

        body = motive.get_rigid_bodies()[body]

        # Load projector's Camera object, created from the calib_projector ratcave_utils CLI tool.
        projector = rc.Camera.from_pickle(projector_filename.encode())

        display = pyglet.window.get_platform().get_default_display()
        screen = display.get_screens()[screen]
        window = LatencyDisplayApp(projector=projector, serial_device=device, fullscreen=True, screen=screen)


        def update_body(dt, window, body):
            motive.update()
            window.dot.position.xyz = body.location
        pyglet.clock.schedule(update_body, window, body)

        pyglet.app.run()



@cli.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.argument('projector_filename', type=click.Path(exists=True))
@click.option('--port', help='Serial Port Name that Latency Device is Connected to.', default='COM7')
@click.option('--screen', help='Screen Number to display on', type=int, default=1)
def latency_body_gen(motive_filename, projector_filename, port, screen):

    with serial.Serial(port=port, timeout=.5) as device:

        device.write('A')

        motive.initialize()
        motive.load_project(motive_filename.encode())
        motive.update()

        # Load projector's Camera object, created from the calib_projector ratcave_utils CLI tool.
        projector = rc.Camera.from_pickle(projector_filename.encode())

        display = pyglet.window.get_platform().get_default_display()
        screen = display.get_screens()[screen]
        window = LatencyDisplayApp(projector=projector, serial_device=device, fullscreen=True, screen=screen)


        def update_body(dt, window, body):
            motive.update()
            window.dot.position.xyz = body.location
        pyglet.clock.schedule(update_body, window, body)



        pyglet.app.run()