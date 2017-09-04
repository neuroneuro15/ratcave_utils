import click
import pyglet
import ratcave as rc
import numpy as np
from itertools import product
from . import cli


def gen_spheres(scale=.01, color=(1., 1., 1.)):
    reader = rc.WavefrontReader(rc.resources.obj_primitives)
    spheres = []
    for row, col in product(*[np.linspace(-.5, .5, 20)]*2):
        sphere = reader.get_mesh('Sphere', position=(row, col, -1), scale=scale,)
        sphere.uniforms['flat_shading'] = 1
        sphere.uniforms['diffuse'] = color
        spheres.append(sphere)
    return spheres


def update_silently(_):
    pass

class DotWindow(pyglet.window.Window):

    def __init__(self, screenidx=1, color=(1., 1., 1.), scale=0.09, *args, **kwargs):
        display = pyglet.window.get_platform().get_default_display()
        screen = display.get_screens()[screenidx]
        super(DotWindow, self).__init__(screen=screen, *args, **kwargs)

        spheres = gen_spheres(scale=scale, color=color)
        self.scene = rc.Scene(meshes=spheres, bgColor=(0,)*3)
        cam = self.scene.camera
        self.shader = rc.Shader.from_file(*rc.resources.genShader)
        cam.projection = rc.OrthoProjection(origin='center', coords='relative')

        cam.update()

        pyglet.clock.schedule(update_silently)


    def on_draw(self):
        with self.shader:
            self.scene.draw()




colors = {'white': (1., 1., 1.), 'red': (1., 0., 0.), 'blue': (0., 0., 1.), 'green': (0., 1., 0.)}

@cli.command()
@click.option('--screen', help='Screen Index to display on', type=int, default=1)
@click.option('--color', help='Color of Dots', type=click.Choice(['white', 'red', 'green', 'blue']), default='white')
@click.option('--size', help='Size of Dots', type=float, default=0.01)
def show_dots(screen, color, size):

    """Displays dot pattern on second screen.  Useful for checking that Motive is detecting the dots."""
    window = DotWindow(fullscreen=True, screenidx=screen, color=colors[color], scale=size)
    pyglet.app.run()


if __name__ == '__main__':
    show_dots()