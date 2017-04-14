import numpy as np
from scipy import linalg
from sklearn import mixture
from sklearn.decomposition import PCA, FastICA

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


def cluster_normals(normal_array, min_clusters=4, max_clusters=9):
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

    vertices = np.array([value for value in vertices.values()] + [floor_verts])
    # vertices = {key: np.array(value) for key, value in vertices.items()}
    # vertices[len(vertices)] = np.array(floor_verts)

    norms = np.vstack((wall_normals, floor_normal))
    # norms = {key: np.array(value) for key, value in enumerate(wall_normals)}
    # norms[len(norms)] = np.array(floor_normal)

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


def face_index(vertices):
    """Takes an MxNx3 array and returns a 2D vertices and MxN face_indices arrays"""
    new_verts = []
    face_indices = []
    for wall in vertices:
        face_wall = []
        for vert in wall:
            if new_verts:
                if not np.isclose(vert, new_verts).all(axis=1).any():
                    new_verts.append(vert)
            else:
                    new_verts.append(vert)
            face_index = np.where(np.isclose(vert, new_verts).all(axis=1))[0][0]
            face_wall.append(face_index)
        face_indices.append(face_wall)
    return np.array(new_verts), np.array(face_indices)


def fan_triangulate(indices):
    """Return an array of vertices in triangular order using a fan triangulation algorithm."""
    if len(indices[0]) != 4:
        raise ValueError("Assumes working with a sequence of quad indices")
    new_indices = []
    for face in indices:
        new_indices.extend([face[[0, 2, 3]], face[[0, 3, 1]]])
    return np.array(new_indices)


def meshify_arena(points, n_surfaces=None, ceiling_offset=.05):
    """Returns vertex and normal coordinates for a 3D mesh model from an Nx3 array of points after filtering.

    Args:
        -points (Nx3 Numpy Array): Data to be fit to a model.
        -n_surfaces: If none, many different models with different numbers of surfaces will be compared.

    Returns:
        -vertices
        -normals
    """

    # Remove Obviously Bad Points according to how far away from main cluster they are
    points_f = points[:]

    # Get the normals of the N-Neighborhood around each point, and filter out points with lowish planarity
    normals_f, explained_variances = normal_nearest_neighbors(points_f)

    # Histogram filter: take the 70% best-planar data to model.
    ll = explained_variances[:, 2]
    normfilter = ll < np.sort(ll)[int(len(ll) * .7)]
    points_ff = points_f[normfilter, :]
    normals_ff = normals_f[normfilter, :]

    # Fit the filtered normal data using a gaussian classifier.
    model = cluster_normals(normals_ff,
                            min_clusters=n_surfaces if n_surfaces else 4,
                            max_clusters=n_surfaces + 1 if n_surfaces else 15)

    # Get normals from model means
    surface_normals = model.means_  # n_components x 3 normals array, giving mean normal for each surface.

    # Calculate mean offset of vertices for each wall
    ids = model.predict(normals_ff)  # index for each point, giving the wall id number (0:n_components)

    #
    surface_offsets = np.array([np.nanmean(points_ff[ids == idx], axis=0) for idx in range(ids.max() + 1)])
    # surface_offsets = np.zeros_like(surface_normals)
    # for idx, _ in enumerate(surface_normals):
    #
    #     surface_offsets[idx, :] = np.mean(points_ff[ids==idx, :], axis=0)
    # assert not np.isnan(surface_offsets.sum()), "Incorrect model: No Points found to assign to at least one wall for intersection calculation."


    ## CALCULATE PLANE INTERSECTIONS TO GET VERTICES ##
    vertices, normals = get_vertices_at_intersections(surface_normals, surface_offsets, points_ff[:, 1].max() + ceiling_offset)

    return vertices, normals


def rotate_to_var(markers):
    """Returns degrees to rotate about y axis so greatest marker variance points in +X direction"""

    # Mean-Center
    markers -= np.mean(markers, axis=0)

    # Vector in direction of greatest variance
    # pca = PCA(n_components=2).fit(markers[:, [0, 2]])
    pca = FastICA(n_components=2).fit(markers[:, [0, 2]])
    coeff_vec = pca.components_[0]

    # Flip coeff_vec in direction of max variance along the vector.
    markers_rotated = pca.fit_transform(markers)  # Rotate data to PCA axes.
    markers_reordered = markers_rotated[markers_rotated[:,0].argsort(), :]  # Reorder Markers to go along first component
    winlen = int(markers_reordered.shape[0]/2+1)  # Window length for moving mean (two steps, with slight overlap)
    var_means = [np.var(markers_reordered[:winlen, 1]), np.var(markers_reordered[-winlen:, 1])] # Compute variance for each half
    coeff_vec = coeff_vec * -1 if np.diff(var_means)[0] < 0 else coeff_vec  # Flip or not depending on which half if bigger.

    # Rotation amount, in radianss
    base_vec = np.array([1, 0])  # Vector in +X direction
    msin, mcos = np.cross(coeff_vec, base_vec), np.dot(coeff_vec, base_vec)
    angle = np.degrees(np.arctan2(msin, mcos))
    print("Angle within function: {}".format(angle))
    return angle


def find_rotation_matrix(points1, points2):
    """
    Returns rotation between two sets of NxM points, which have been rotated from one another.

    Parameters
    ----------
    points1: MxN (NPoints x NDimensions) matrix
    points2: MxN (NPoints x NDimensions) matrix, which is 'points1' rotated by some amount.

    Examples
    --------
    >>> find_rotation_matrix([[1., 0.], [0., .5]], [[0., 1.], [-.5, 0.]])
    array([[ 0., -1.],
           [ 1.,  0.]])

    >>> find_rotation_matrix([[1.00001, 0.001], [0.0001, .50001]], [[0., 1.], [-.5, 0.]])
    array([[-0., -1.],
           [ 1.,  0.]])

    >>> find_rotation_matrix([[1., 0.], [0., .5]], [[-1., 0.], [0., -.5]])
    array([[-1.,  0.],
           [ 0., -1.]])

    >>> find_rotation_matrix([[1., 0, 0], [0, .5, 2], [-.3, .4, 0]], [[0.,  0, -1], [2, 0.5, 0],[-0, .4, .3]])
    array([[-0., -0.,  1.],
           [ 0.,  1., -0.],
           [-1.,  0., -0.]])
    """
    p1 = np.array(points1)
    p2 = np.array(points2)

    if p1.shape != p2.shape:
        raise ValueError("Both Matrices must be the same size (M Points x N Dimensions)")
    if p1.shape[0] < p1.shape[1] or p2.shape[0] < p1.shape[1]:
        raise ValueError("Underranked Matrices.  Need at least as many points as spatial dimensions.")

    rotmat = np.dot(linalg.pinv(p2), p1)

    return rotmat