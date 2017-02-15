import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA


def rotate_to_var(markers):
    """Returns degrees to rotate about y axis so greatest marker variance points in +X direction"""

    # Mean-Center
    markers -= np.mean(markers, axis=0)

    # Vector in direction of greatest variance
    pca = PCA(n_components=2).fit(markers[:, [0, 2]])
    coeff_vec = pca.components_[0]

    # Flip coeff_vec in direction of max variance along the vector.
    markers_rotated = pca.fit_transform(markers)  # Rotate data to PCA axes.
    markers_reordered = markers_rotated[markers_rotated[:,0].argsort(), :]  # Reorder Markers to go along first component
    winlen = int(markers_reordered.shape[0]/2+1)  # Window length for moving mean (two steps, with slight overlap)
    var_means = [np.var(markers_reordered[:winlen, 1]), np.var(markers_reordered[-winlen:, 1])] # Compute variance for each half
    coeff_vec = coeff_vec * -1 if np.diff(var_means)[0] < 0 else coeff_vec  # Flip or not depending on which half if bigger.

    # Rotation amount, in radians
    base_vec = np.array([1, 0])  # Vector in +X direction
    msin, mcos = np.cross(coeff_vec, base_vec), np.dot(coeff_vec, base_vec)
    angle = np.degrees(np.arctan2(msin, mcos))
    print("Angle within function: {}".format(angle))
    return angle


def correct_orientation_motivepy(rb, n_attempts=3):
    import motive
    """Reset the orientation to account for between-session arena shifts"""
    for attempt in range(n_attempts):
            rb.reset_orientation()
            motive.update()
    additional_rotation = rotate_to_var(np.array(rb.point_cloud_markers))
    return additional_rotation


def correct_orientation_natnet(rb, n_attempts=3):
    """Assumes the orientation is reset already (need MotivePy to do it automatically) to account for between-session arena shifts"""
    print(("Warning: Assuming that the orientation has been reset to 0,0,0 for the {} rigid body".format(rb.name)))
    additional_rotation = rotate_to_var(np.array([m.position for m in rb.markers]))
    return additional_rotation


def update_world_position_motivepy(meshes, arena_rb, additional_rot_y_rotation):
    """# Update the positions of everything, based on the MotivePy data of the arena rigid body"""
    for mesh in meshes:
        mesh.world.position = arena_rb.location
        mesh.world.rotation = arena_rb.rotation_global
        mesh.world.rot_y += additional_rot_y_rotation
    return


def update_world_position_natnet(meshes, arena_rb, additional_rot_y_rotation):
    """# Update the positions of everything, based on the MotivePy data of the arena rigid body"""
    for mesh in meshes:
        mesh.world.position = arena_rb.position
        mesh.world.rotation = arena_rb.rotation
        mesh.world.rot_y += additional_rot_y_rotation
    return


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
