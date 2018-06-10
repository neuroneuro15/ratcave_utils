import click
import pyglet
import pyglet.gl as gl
import ratcave as rc
import numpy as np
from itertools import product, chain
from . import cli


colors = {'white': (255, 255, 255), 'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0)}


class DotWindow(pyglet.window.Window):

    def __init__(self, screenidx=1, color=(255, 255, 255), size=30, *args, **kwargs):
        display = pyglet.window.get_platform().get_default_display()
        screen = display.get_screens()[screenidx]
        super(DotWindow, self).__init__(screen=screen, *args, **kwargs)
        gl.glPointSize(size)
        gl.glEnable(gl.GL_POINT_SMOOTH)

        self.color = color
        
        n_cols = 30
        self.coords = np.array(tuple(c for el in product(*[range(0, self.width, int(self.width / n_cols))] * 2) for c in el)) + int(self.width / n_cols / 2)
        
        pyglet.clock.schedule(lambda x: x)


    def on_draw(self):
        self.clear()
        n_points = int(len(self.coords) / 2)
        pyglet.graphics.draw(n_points, gl.GL_POINTS, ('v2i', self.coords), ('c3B', self.color * n_points))
    

def main(screen=1, color='white', size=20):
    """Displays dot pattern on second screen.  Useful for checking that Motive is detecting the dots."""
    window = DotWindow(fullscreen=True, screenidx=screen, color=colors[color], size=size)
    pyglet.app.run()	
	

@cli.command()
@click.option('--screen', help='Screen Index to display on', type=int, default=1)
@click.option('--color', help='Color of Dots', type=click.Choice(['white', 'red', 'green', 'blue']), default='white')
@click.option('--size', help='Size of Dots', type=float, default=20)
def show_dots(screen, color, size):
    main(screen=screen, color=color, size=size)
    

if __name__ == '__main__':
    main()