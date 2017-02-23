import click
import pyglet
import ratcave as rc
from . import cli


@cli.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.argument('projector_filename', type=click.Path(exists=True))
@click.argument('arena_filename', type=click.Path(exists=True))
def view_arenafit(motive_filename, projector_filename, arena_filename):
    # """Displays mesh in .obj file.  Useful for checking that files are rendering properly."""

    reader = rc.WavefrontReader(arena_filename)
    arena = reader.get_mesh('Arena')
    arena.rotation = arena.rotation.to_quaternion()
    print('Arena Loaded. Position: {}, Rotation: {}'.format(arena.position, arena.rotation))

    camera = rc.Camera.from_pickle(projector_filename)
    light = rc.Light(position=(camera.position.xyz))

    root = rc.EmptyEntity()
    root.add_child(arena)
    scene = rc.Scene(meshes=root,
                     camera=camera,
                     light=light,
                     bgColor=(.2, .4, .2))
    scene.gl_states = scene.gl_states[:-1]

    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[0]
    window = pyglet.window.Window(fullscreen=True, screen=screen)

    label = pyglet.text.Label()

    @window.event
    def on_draw():
        with rc.resources.genShader:
            scene.draw()
        label.draw()

    @window.event
    def on_resize(width, height):
        camera.projection.aspect = float(width) / height


    import motive
    motive.initialize()
    motive.load_project(motive_filename.encode())
    motive.update()
    rb = motive.get_rigid_bodies()['Arena']

    @window.event
    def update_arena_position(dt):
        motive.update()
        arena.position.xyz = rb.location
        arena.rotation.xyzw = rb.rotation_quats
        label.text = "({:2f}, {:2f}, {:2f}), ({:2f}, {:2f}, {:2f})".format(*(arena.position.xyz + rb.location))
    pyglet.clock.schedule(update_arena_position)

    def spin_arena(dt):
        root.rotation.y += 10 * dt
    pyglet.clock.schedule(spin_arena)

    pyglet.app.run()


if __name__ == '__main__':
    view_arenafit()