import pyglet
pyglet.options['debug_gl'] = False
import ratcave as rc
import numpy as np

#Inputs
obj_name = 'Cube'
num_objects = 100
use_cubemap = True
has_uniforms = True

# Create Window
window = pyglet.window.Window(resizable=True, vsync=False)

# Generate Objects
reader = rc.WavefrontReader(rc.resources.obj_primitives)

player = rc.mesh.EmptyMesh()
screen = reader.get_mesh('Plane')
screen.position = 0, 0, -1
screen.scale = .7
screen.uniforms['diffuse'] = .7, 0, 0

def sphere_factory(reader, n=10):
    for _ in range(n):
        sphere = reader.get_mesh(obj_name, scale=.1)
        sphere.position = np.append(2 * np.random.random(2) - 1, [-2])
        if not has_uniforms:
            sphere.uniforms = rc.UniformCollection()
        else:
            sphere.uniforms['diffuse']= np.random.random(3)
        sphere.update()
        yield sphere

spheres = [sphere for sphere in sphere_factory(reader, num_objects)]


# Debug Text
label_fps = pyglet.text.Label('FPS', font_size=20)

# Create Scenes and FBOs to render onto
proj_scene = rc.Scene(meshes=spheres)
proj_scene.camera.aspect = 1.
proj_scene.camera.fov_y = 90.

god_scene = rc.Scene(meshes=[player, screen])
cubetexture = rc.TextureCube()
screen.texture = cubetexture
fbo_cube = rc.FBO(cubetexture)

# Run Display and Update Loops
@window.event
def on_resize(width, height):
    # proj_scene.camera.aspect = width / float(height)
    god_scene.camera.aspect = width / float(height)

@window.event
def on_draw():
    # proj_scene.draw()
    window.clear()
    if use_cubemap:
        with fbo_cube:
            proj_scene.draw360_to_texture(cubetexture)
        god_scene.draw()
    else:
        proj_scene.draw()
    label_fps.draw()


def update(dt):
    for sphere in spheres:
        sphere.rot_y += 45 * dt
    fps_text = 'msecs: {}\nFPS: {}'.format(dt * 1000, 1. / (dt + .000001))
    label_fps.text = fps_text

pyglet.clock.schedule(update)


pyglet.app.run()


