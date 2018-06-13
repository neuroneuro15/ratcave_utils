
import pyglet
from natnetclient import NatClient
import click
import ratcave as rc
from . import cli
import pyglet.gl as gl


@cli.command()
@click.argument('projector_filename', type=click.Path(exists=True))
@click.argument('arena_filename', type=click.Path(exists=True))
@click.option('--body', help='Name of rigidBody to track')
@click.option('--screen', help='Screen Number to display on', type=int, default=1)
def vr_demo(projector_filename, arena_filename, body, screen):

    client = NatClient()
    arena_body = client.rigid_bodies['Arena']
    subject = client.rigid_bodies[body]

    # Load projector's Camera object, created from the calib_projector ratcave_utils CLI tool.
    config = gl.Config(double_buffer=True, stereo=True)
    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[screen]
    window = pyglet.window.Window(fullscreen=True, screen=screen, vsync=False, config=config)


    fbo = rc.FBO(rc.TextureCube(width=4096, height=4096))

    shader3d = rc.resources.cube_shader

    reader = rc.WavefrontReader(rc.resources.obj_primitives)
    mesh = reader.get_mesh('Monkey', scale=.7)
    mesh.position.y = .1
    mesh.rotation.x = 45
    mesh.rotation.y = 180
    mesh.uniforms['ambient'] = .15, .15, .15
    # mesh.rotation = mesh.rotation.to_quaternion()

    arena = rc.WavefrontReader(arena_filename.encode()).get_mesh('Arena')
    arena.rotation = arena.rotation.to_quaternion()
    arena.textures.append(fbo.texture)

    vr_scene = rc.Scene(meshes=[mesh], bgColor=(0.1, .1, 0.3))
    vr_camgroup = rc.StereoCameraGroup(distance=.05)
    vr_camgroup.rotation = vr_camgroup.rotation.to_quaternion()
    vr_scene.camera = rc.Camera(projection=rc.PerspectiveProjection(fov_y=90, aspect=1., z_near=.005, z_far=3.))


    scene = rc.Scene(meshes=[arena], bgColor=(0., 0., .3))
    scene.gl_states.states = scene.gl_states.states[:-1]

    camera = rc.Camera.from_pickle(projector_filename.encode())
    print(camera.position)
    print(camera.rotation)
    print(camera.orientation)
    scene.camera = camera
    scene.light.position.xyz = camera.position.xyz
    vr_scene.light.position.xyz = camera.position.xyz

    @window.event
    def on_draw():
        with shader3d:

            gl.glDrawBuffer(gl.GL_BACK_LEFT)
            with fbo:

                    # gl.glClear(gl.GL_DEPTH_BUFFER_BIT)


                # gl.glColorMask(True, False, False, True)
                window.clear()
                vr_scene.camera.position.xyz = vr_camgroup.left.position_global
                vr_scene.camera.projection.match_aspect_to_viewport()
                vr_scene.draw360_to_texture(fbo.texture)
            gl.glColorMask(True, True, True, True)
            scene.camera.projection.match_aspect_to_viewport()
            scene.draw()


            gl.glDrawBuffer(gl.GL_BACK_RIGHT)
            with fbo:

                # gl.glColorMask(False, True, True, True)

                vr_scene.camera.position.xyz = vr_camgroup.right.position_global
                vr_scene.camera.projection.match_aspect_to_viewport()
                vr_scene.draw360_to_texture(fbo.texture)

            # gl.glColorMask(True, True, True, True)
            scene.camera.projection.match_aspect_to_viewport()
            scene.draw()



    def update_body(dt, body):
        arena.position.xyz = arena_body.position
        arena.rotation.xyzw = arena_body.quaternion
        mesh.position.xz = arena.position.xz
        # mesh.position.y -= .07
        vr_camgroup.position.xyz = subject.position
        vr_camgroup.rotation.xyzw = subject.quaternion
        scene.camera.uniforms['playerPos'] = subject.position

        # mesh.rotation.y += 10. * dt

    pyglet.clock.schedule(update_body, body)

    pyglet.app.run()


