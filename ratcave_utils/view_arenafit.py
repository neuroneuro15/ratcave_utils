import click
import pyglet
import ratcave as rc
from . import cli
from natnetclient import NatClient
from pyglet.window import key, FPSDisplay
import pickle

@cli.command()
@click.argument('projector_filename', type=click.Path(exists=True))
@click.argument('arena_filename', type=click.Path(exists=True))
@click.option('--screen', type=int, help='Screen Number for Window to Appear On', default=1)
def view_arenafit(projector_filename, arena_filename, screen):
    # """Displays mesh in .obj file.  Useful for checking that files are rendering properly."""

    reader = rc.WavefrontReader(arena_filename)
    arena = reader.get_mesh('Arena', mean_center=True)
    arena.rotation = arena.rotation.to_quaternion()
    print('Arena Loaded. Position: {}, Rotation: {}'.format(arena.position, arena.rotation))

    with open(projector_filename) as f:
        camera = pickle.load(f)
    camera.reset_uniforms()
    # camera.projection.fov_y = 39
    # camera.projection.z_far = 10.
    light = rc.Light(position=(camera.position.xyz))

    root = rc.EmptyEntity()
    root.add_child(arena)

    sphere = rc.WavefrontReader(rc.resources.obj_primitives).get_mesh('Sphere', scale=.05)
    root.add_child(sphere)

    scene = rc.Scene(meshes=root,
                     camera=camera,
                     light=light,
                     bgColor=(.2, .4, .2))
    scene.gl_states.states = scene.gl_states.states[:-1]

    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[screen]
    window = pyglet.window.Window(fullscreen=True, screen=screen)

    label = pyglet.text.Label()
    fps_display = FPSDisplay(window)

    @window.event
    def on_draw():
        with rc.default_shader:
            scene.draw()
        # label.draw()
        # window.clear()
        fps_display.draw()

    @window.event
    def on_resize(width, height):
        camera.projection.aspect = float(width) / height

    @window.event
    def on_key_release(sym, mod):
        if sym == key.UP:
            scene.camera.projection.fov_y += .1
        elif sym == key.DOWN:
            scene.camera.projection.fov_y -= .1
        elif sym == key.LEFT:
            scene.camera.rotation.z += .1
        elif sym == key.RIGHT:
            scene.camera.rotation.z -= .1
        elif sym == key.A:
            scene.camera.rotation.y -= .005
        elif sym == key.D:
            scene.camera.rotation.y += .005
        elif sym == key.W:
            scene.camera.rotation.x += .005
        elif sym == key.S:
            scene.camera.rotation.x -= .005


        scene.camera.reset_uniforms()  # TODO: Fix Pickle bug so this isn't needed!
        print(scene.camera.projection.fov_y)
        print(scene.camera.projection.aspect)

    client = NatClient()
    rb = client.rigid_bodies['Arena']

    def update_arena_position(dt):
        arena.position.xyz = rb.position
        arena.rotation.xyzw = rb.quaternion
        arena.update()

        sphere.position.xyz = rb.position
        label.text = "aspect={}, fov_y={}, ({:2f}, {:2f}, {:2f}), ({:2f}, {:2f}, {:2f})".format(scene.camera.projection.aspect,
                                                                                                scene.camera.projection.fov_y,    *(arena.position.xyz + rb.position))
    pyglet.clock.schedule(update_arena_position)

    pyglet.app.run()


if __name__ == '__main__':
    view_arenafit()