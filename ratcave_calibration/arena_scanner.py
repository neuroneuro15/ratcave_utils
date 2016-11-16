__author__ = 'nickdg'

from _transformations import rotation_matrix

from os import path
import click
import motive
import numpy as np
import pyglet
import ratcave as rc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from psychopy import visual
from ratcave import utils
from sklearn import mixture
from sklearn.decomposition import PCA

import filters
import orienting
from plotting import plot_3d
from ratcave_calibration import hardware


np.set_printoptions(precision=3, suppress=True)


class GridScanWindow(pyglet.window.Window):

    def __init__(self, *args, **kwargs):
        """
        Returns Window with everything needed for doing arena scanning.  Will automatically close when all camera movement
        has completed.
        """
        super(GridScanWindow, self).__init__(*args, **kwargs)

        wavefront_reader = rc.WavefrontReader(rc.resources.obj_primitives)
        self.mesh = wavefront_reader.get_mesh('Grid', position=[0., 0., -1.], scale=1.5, point_size=12, drawstyle='point')
        self.mesh.uniforms['diffuse'] = [1., 1., 1.]  # Make white
        self.mesh.uniforms['flat_shading'] = True

        self.scene = rc.Scene([self.mesh], bgColor=(0, 0, 0))
        self.scene.camera.ortho_mode = True

        self.cam_positions = self.gen_camera_positions()

        self.marker_pos = []
        pyglet.clock.schedule(self.detect_projection_point)
        pyglet.clock.schedule(self.move_camera)

    def gen_camera_positions(self, pointwidth=0.06):
        """Generator that returns the next camera position, which moves in a circle"""
        for theta in np.linspace(0, 2*np.pi, 40)[:-1]:
            yield (pointwidth * np.sin(theta)), (pointwidth * np.cos(theta)), -1

    def move_camera(self, dt):
        """Randomly moves the mesh center to somewhere between xlim and ylim"""
        try:
            self.scene.camera.position = next(self.cam_positions)
        except StopIteration:
            print("End of Camera Position list reached. Closing window...")
            pyglet.clock.unschedule(self.detect_projection_point)
            self.close()

    def on_draw(self):
        """Render the scene!"""
        self.scene.draw()

    def detect_projection_point(self, dt):
        """Use Motive to detect the projected mesh in 3D space"""
        motive.flush_camera_queues()
        for el in range(2):
            motive.update()

        markers = motive.get_unident_markers()
        markers = [marker for marker in markers if 0.08 < marker[1] < 0.50]

        click.echo("{} markers detected.".format(len(markers)))
        self.marker_pos.extend(markers)


@click.command()
@click.argument('motive_filename', type=click.Path(exists=True))
@click.argument('output_filename', type=click.Path())
@click.option('--body', help='Name of arena rigidbody to track', default='arena')
@click.option('--nomeancenter', help='Flag: Skips mean-centering of arena.', type=bool, default=False)
@click.option('--nopca', help='Flag: skips PCA-based arena marker rotation (used for aligning IR markers to image points)', type=bool, default=False)
@click.option('--nsides', help='Number of Arena Sides.  If not given, will be estimated from data.', type=int, default=0)
def run(motive_filename, output_filename, body, nomeancenter, nopca, nsides):
    """Runs Arena Scanning algorithm."""

    output_filename = output_filename + '.obj' if not path.splitext(output_filename)[1] else output_filename
    assert path.splitext(output_filename)[1] == '.obj', "Output arena filename must be a Wavefront (.obj) file"

    # Load Motive Project File
    motive_filename = motive_filename.encode()
    motive.load_project(motive_filename)
    hardware.motive_camera_vislight_configure()
    motive.update()

    # Get Arena's Rigid Body
    rigid_bodies = motive.get_rigid_bodies()
    assert body in rigid_bodies, "RigidBody {} not found in project file.  Available body names: {}".format(body, list(rigid_bodies.keys()))
    assert len(rigid_bodies[body].markers) > 5, "At least 6 markers in the arena's rigid body is required. Only {} found".format(len(rigid_bodies[body].markers))


    # TODO: Fix bug that requires scanning be done in original orientation (doesn't affect later recreation, luckily.)
    for attempt in range(3):  # Sometimes it doesn't work on the first try, for some reason.
        rigid_bodies[body].reset_orientation()
        motive.update()
        if sum(np.abs(rigid_bodies[body].rotation)) < 1.:
            break
    else:
        raise ValueError("Rigid Body Orientation not Resetting to 0,0,0 after 3 attempts.  This happens sometimes (bug), please just run the script again.")

    # Scan points
    display = pyglet.window.get_platform().get_default_display()
    screen = display.get_screens()[1]
    window = GridScanWindow(screen=screen, fullscreen=True)
    pyglet.app.run()
    points = np.array(window.marker_pos)
    assert(len(points) > 100), "Only {} points detected.  Tracker is not detecting enough points to model.  Is the projector turned on?".format(len(points))
    assert points.ndim == 2

    # Rotate all points to be mean-centered and aligned to Optitrack Markers direction or largest variance.
    markers = np.array(rigid_bodies[body].point_cloud_markers)
    if not nomeancenter:
        points -= np.mean(markers, axis=0)
    if not nopca:
        points = np.dot(points, rotation_matrix(np.radians(orienting.rotate_to_var(markers)), [0, 1, 0])[:3, :3])

    # Plot preview of data collected
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points.T)
    plt.show()

    # Get vertex positions and normal directions from the collected data.
    vertices, normals = meshify(points, n_surfaces=nsides)
    vertices = {wall: fan_triangulate(reorder_vertices(verts)) for wall, verts in vertices.items()}  # Triangulate

    # Write wavefront .obj file to app data directory and user-specified directory for importing into Blender.
    wave_str = data_to_wavefront(body, vertices, normals)

    # Write to output file
    with open(output_filename, 'wb') as wavfile:
        wavfile.write(wave_str)

    # Show resulting plot with points and model in same place.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points[::12, :].T)
    for idx, verts in vertices.items():
        ax.plot(*np.vstack((verts, verts[0, :])).T)
    plt.show()



def normal_nearest_neighbors(data, n_neighbors=40):
    """Find the normal direction of a hopefully-planar cluster of n_neighbors"""
    from sklearn.neighbors import NearestNeighbors

    # K-Nearest neighbors on whole dataset
    nbrs = NearestNeighbors(n_neighbors).fit(data)
    _, indices = nbrs.kneighbors(data)

    # PCA on each cluster of k-nearest neighbors
    latent_all, normal_all = [], []
    for idx_array in indices:

        pp = PCA(n_components=3).fit(data[idx_array, :])  # Perform PCA

        # Get the percent variance of each component
        latent_all.append(pp.explained_variance_ratio_)

        # Get the normal of the plane along the third component (flip if pointing in -y direction)
        normal = pp.components_[2] if pp.components_[2][1] > 0 else -pp.components_[2]
        normal_all.append(normal)

    # Convert to NumPy Array and return
    return np.array(normal_all), np.array(latent_all)


def cluster_normals(normal_array, min_clusters=4, max_clusters=15):
    """Returns sklearn model from clustering an NxK array, comparing different numbers of clusters for a best fit."""

    model, old_bic = None, 1e32
    for n_components in range(min_clusters, max_clusters):

        gmm = mixture.GMM(n_components=n_components) # Fit the filtered normal data using a gaussian classifier
        temp_model = gmm.fit(normal_array)
        temp_bic = temp_model.bic(normal_array)
        print("N Components: {}\tBIC: {}".format(n_components, temp_bic))
        model, old_bic = (temp_model, temp_bic) if temp_bic < old_bic else (model, old_bic)

    return model


def get_vertices_at_intersections(normals, offsets, ceiling_height):
    """Returns a dict of vertices and normals for each surface intersecton of walls given by the Nx3 arrays of
    normals and offsets."""

    from scipy import spatial

    # Calculate d in equation ax + by + cz = d
    dd = np.sum(normals * offsets, axis=1)

    # Automatically Separate out the floor from the walls.
    floor_idx = normals[:, 1].argsort()[-1]
    wall_normals, wall_d = np.delete(normals, floor_idx, axis=0), np.delete(dd, floor_idx)
    floor_normal, floor_d = normals[floor_idx, :], dd[floor_idx]

    # Get neighbors between all walls (excluding the floor, which touches everything.)
    distances = spatial.distance_matrix(wall_normals, wall_normals) + (3 * np.eye(wall_normals.shape[0]))
    neighboring_walls = np.sort(distances.argsort()[:, :2])  # Get the two closest wall indices to each wall
    neighbors =  {dd: el.tolist() for (dd, el) in enumerate(neighboring_walls)}

    # Solve for intersection between the floor/ceiling and adjacent walls,
    vertices = {wall: [] for wall in range(len(neighbors))}
    floor_verts = []
    for wall in neighbors:
        for adj_wall in neighbors[wall]:
            for normal, d in ((floor_normal, floor_d), (np.array([0., 1., 0.]), ceiling_height)):
                all_norms = np.vstack((wall_normals[wall], wall_normals[adj_wall], normal))
                all_d = np.array((wall_d[wall], wall_d[adj_wall], d))
                vertex = np.linalg.solve(all_norms, all_d).transpose()
                vertices[wall].append(vertex)

                if d < ceiling_height and vertex.tolist() not in floor_verts:
                    floor_verts.append(vertex.tolist())

    # Convert vertex lists to dict of NumPy arrays
    vertices = {key: np.array(value) for key, value in vertices.items()}
    vertices[len(vertices)] = np.array(floor_verts)

    norms = {key: np.array(value) for key, value in enumerate(wall_normals)}
    norms[len(norms)] = np.array(floor_normal)

    return vertices, norms


def reorder_vertices(vertices):
    """Takes an unordered Nx3 vertex array and reorders them so the resulting face's normal vector faces upwards."""

    # Turn the vertex positions to unit-length rays from the mean position (assumes coplanarity)
    vertices = np.array(vertices)
    rays = vertices - np.mean(vertices, axis=0)
    rays /= np.linalg.norm(rays, axis=1).reshape(-1, 1)  # Normalize their lengths, so we get pure cos and sin values

    # Build a covariance matrix, which is the cos values
    cov = np.arccos(np.dot(rays, rays.T) - np.eye(len(rays)))

    # Compare the cross product of each ray combination to the normal, and only keep if the same direction.
    cross_mask = np.zeros_like(cov, dtype=bool)
    for i, ray_i in enumerate(rays):
        for j, ray_j in enumerate(rays):
            cp = np.cross(ray_i, ray_j)
            cross_mask[i, j] = np.dot(cp, [0, 1, 0]) > 0.

    # Apply the filter and reorder the vertices
    cov_filtered = cov * cross_mask
    cov_filtered[cov_filtered==0] = 100.  # Change zeros to a large number, so they aren't taken as the min value.
    new_indices = cov_filtered.argsort()[:,0]

    nn_i, idx = [], 0
    for _ in new_indices:
        nn_i.append(new_indices[idx])
        idx = nn_i[-1]

    return vertices[nn_i, :]


def fan_triangulate(vertices):
    """Return an array of vertices in triangular order using a fan triangulation algorithm."""
    return np.array([el for (ii, jj) in zip(vertices[1:-1], vertices[2:]) for el in [vertices[0], ii, jj]])


def data_to_wavefront(mesh_name, vert_dict, normal_dict):
    """Returns a wavefront .obj string using pre-triangulated vertex dict and normal dict as reference."""

    # Put header in string
    wavefront_str = "# Blender v2.69 (sub 5) OBJ File: ''\n" + "# www.blender.org\n" + "o {name}\n".format(name=mesh_name)

    # Write Vertex data from vert_dict
    for wall in vert_dict:
        for vert in vert_dict[wall]:
            wavefront_str += "v {0} {1} {2}\n".format(*vert)

    # Write (false) UV Texture data
    wavefront_str += "vt 1.0 1.0\n"

    # Write Normal data from normal_dict
    for wall, norm in normal_dict.items():
        wavefront_str += "vn {0} {1} {2}\n".format(*norm)

    # Write Face Indices (1-indexed)
    vert_idx = 0
    for wall in vert_dict:
        for _ in range(0, len(vert_dict[wall]), 3):
            wavefront_str += 'f '
            for vert in range(3): # 3 vertices in each face
                vert_idx += 1
                wavefront_str += "{v}/1/{n} ".format(v=vert_idx, n=wall+1)
            wavefront_str = wavefront_str[:-1] + '\n'  # Cutoff trailing space and add a newline.

    # Return Wavefront string
    return wavefront_str


def meshify(points, n_surfaces=None):
    """Returns vertex and normal coordinates for a 3D mesh model from an Nx3 array of points after filtering.

    Args:
        -points (Nx3 Numpy Array): Data to be fit to a model.
        -n_surfaces: If none, many different models with different numbers of surfaces will be compared.

    Returns:
        -vertices
        -normals
    """

    # Remove Obviously Bad Points according to how far away from main cluster they are
    histmask = np.ones(points.shape[0], dtype=bool)  # Initializing mask with all True values
    for coord in range(3):
        histmask &= filters.hist_mask(points[:, coord], keep='middle')
    points_f = points[histmask, :]

    # Get the normals of the N-Neighborhood around each point, and filter out points with lowish planarity
    normals_f, explained_variances = normal_nearest_neighbors(points_f)

    # Histogram filter: take the 70% best-planar data to model.
    normfilter = filters.hist_mask(explained_variances[:, 2], threshold=.7, keep='lower')
    points_ff = points_f[normfilter, :]
    normals_ff = normals_f[normfilter, :]

    ceiling_height = points_ff[:, 1].max() + .005

    # Fit the filtered normal data using a gaussian classifier.
    min_clusters = n_surfaces if n_surfaces else 4
    max_clusters = n_surfaces + 1 if n_surfaces else 15
    model = cluster_normals(normals_ff, min_clusters=min_clusters, max_clusters=max_clusters)

    # Get normals from model means
    surface_normals = model.means_  # n_components x 3 normals array, giving mean normal for each surface.

    # Calculate mean offset of vertices for each wall
    ids = model.predict(normals_ff)  # index for each point, giving the wall id number (0:n_components)

    surface_offsets = np.zeros_like(surface_normals)
    for idx in range(len(surface_normals)):
        surface_offsets[idx, :] = np.mean(points_ff[ids==idx, :], axis=0)

    assert not np.isnan(surface_offsets.sum()), "Incorrect model: No Points found to assign to at least one wall for intersection calculation."

    ## CALCULATE PLANE INTERSECTIONS TO GET VERTICES ##
    vertices, normals = get_vertices_at_intersections(surface_normals, surface_offsets, ceiling_height)
    return vertices, normals


if __name__ == '__main__':
    run()