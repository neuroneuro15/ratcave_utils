import click
import pyglet
import ratcave as rc
from . import cli


@cli.command()
@click.argument('body')
@click.option('--obj_filename', type=click.Path(exists=True), default=rc.resources.obj_primitives, help='Obj Filename to load')
def view_mesh(body, obj_filename):
    # """Displays mesh in .obj file.  Useful for checking that files are rendering properly."""

    reader = rc.WavefrontReader(obj_filename)
    mesh = reader.get_mesh(body, position=(0, 0, -1))
    print(mesh.vertices.shape)

    mesh.scale.x = .1 / mesh.vertices.mean(axis=0).max()
    camera = rc.Camera(projection=rc.PerspectiveProjection(fov_y=20))
    light = rc.Light(position=(camera.position.xyz))

    scene = rc.Scene(meshes=[mesh],
                     camera=camera,
                     light=light,
                     bgColor=(.2, .4, .2))
    scene.gl_states = scene.gl_states[:-1]

    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[0]
    window = pyglet.window.Window(fullscreen=True, screen=screen)

    fbo = rc.FBO(rc.Texture(width=4096, height=4096))
    quad = rc.gen_fullscreen_quad()
    quad.texture = fbo.texture

    @window.event()
    def on_draw():
        with rc.resources.genShader, fbo:
            scene.draw()
        with rc.resources.deferredShader:
            quad.draw()

    @window.event()
    def on_resize(width, height):
        camera.projection.aspect = float(width) / height

    @window.event()
    def on_mouse_motion(x, y, dx, dy):
        x, y = x / float(window.width) - .5, y / float(window.height) - .5
        mesh.rotation.x = -360 * y
        mesh.rotation.y = 360 * x


    pyglet.app.run()


if __name__ == '__main__':
    view_mesh()