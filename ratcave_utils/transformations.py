import wavefront_reader
import numpy as np
from sklearn.decomposition import PCA


def arena_obj_to_transformation(objfile, object="Arena"):
    """Returns the 3x3 rotation matrix and 1x3 translation array for the "object" object in the objfile, using PCA."""
    verts = wavefront_reader.read_objfile(objfile)[object]['v']
    uverts = np.unique(verts, axis=0)
    floor_verts = uverts[uverts[:,1].argsort()][:4, :]
    pca = PCA(n_components=3).fit(floor_verts)
    rotmat, translation = pca.components_, pca.mean_
    return rotmat, translation
