import click
import pyglet
import ratcave as rc
import numpy as np
from itertools import product
from . import cli


def gen_spheres(scale=.01):
    reader = rc.WavefrontReader(rc.resources.obj_primitives)
    spheres = []
    for row, col in product(*[np.linspace(-1, 1, 20)]*2):
        spheres.append(reader.get_mesh('Sphere', position=(row, col, -1), scale=scale,))
        spheres[-1].uniforms['flat_shading'] = True
    return spheres


def update_silently(_):
    pass

class DotWindow(pyglet.window.Window):

    def __init__(self, screenidx=1, *args, **kwargs):
        display = pyglet.window.get_platform().get_default_display()
        screen = display.get_screens()[screenidx]
        super(DotWindow, self).__init__(screen=screen, *args, **kwargs)

        spheres = gen_spheres()
        self.scene = rc.Scene(meshes=spheres, bgColor=(0,)*3)
        self.scene.camera.ortho_mode = True
        self.scene.camera._update_projection_matrix()
        self.scene.camera.update()

        pyglet.clock.schedule(update_silently)


    def on_draw(self):
        self.scene.draw()






@cli.command()
@click.option('--screen', help='Screen Index to display on', type=int, default=1)
def show_dots(screen):
    """Displays dot pattern on second screen.  Useful for checking that Motive is detecting the dots."""
    window = DotWindow(fullscreen=True, screenidx=screen)
    pyglet.app.run()


if __name__ == '__main__':
    show_dots()